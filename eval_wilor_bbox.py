"""Evaluate the WiLoR YOLO hand detector on the WHIM test set, with the SAME
metrics as eval_detector_bbox.py so it's directly comparable to the MediaPipe
detector (pretrained + WHIM-finetuned).

WiLoR's detector outputs real hand boxes (xyxy px) + handedness class + conf
(see WiLoR/demo.py). We treat it as a single "hand" class for box detection.

  .venv-wilor/bin/python eval_wilor_bbox.py [--limit N] [--out eval_wilor.json]

AP@[.50:.95]/AP50/AP75 are computed over the full score range (conf>=--min-conf,
default 0.001). precision/recall/center-recall/coverage are reported at an
operating point (--op-conf, default 0.5, matching the MediaPipe eval).
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from eval_detector_bbox import average_precision, iou_xyxy  # reuse exact metrics

TEST_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/test/anno"
DETECTOR = "/media/vimal/T7_2TB/best_hand_detector/WiLoR/pretrained_models/detector.pt"


def clamp(b, w, h):
    return [min(max(b[0], 0.0), w), min(max(b[1], 0.0), h),
            min(max(b[2], 0.0), w), min(max(b[3], 0.0), h)]


def load_gt(npy, w, h):
    out = []
    for hd in np.load(npy, allow_pickle=True):
        b = clamp([float(v) for v in np.asarray(hd["bbox"], float)], w, h)
        if b[2] > b[0] and b[3] > b[1]:
            out.append(b)
    return out


def coverage_and_center(preds_by_img, gts_by_img):
    cov, center_hit, n = [], 0, 0
    for img, gts in gts_by_img.items():
        pl = [b for _, b in preds_by_img.get(img, [])]
        for gb in gts:
            n += 1
            cov.append(max((iou_xyxy(p, gb) for p in pl), default=0.0))
            cx, cy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
            if any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in pl):
                center_hit += 1
    return (float(np.mean(cov)) if cov else 0.0), center_hit / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-conf", type=float, default=0.001, help="for AP curve")
    ap.add_argument("--op-conf", type=float, default=0.5, help="operating point for P/R")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="eval_wilor_bbox.json")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(TEST_ROOT, "*", "*.npy")))
    if args.limit:
        frames = frames[: args.limit]
    jpgs = [f[:-4] + ".jpg" for f in frames]
    print(f"WiLoR detector on {len(frames)} frames (device={args.device}) ...", flush=True)

    det = YOLO(DETECTOR)
    det.to(args.device)

    preds_all, gts_by_img = {}, {}      # preds_all: img -> [(score, box)]
    t0 = time.time()
    for start in range(0, len(frames), args.batch):
        chunk = jpgs[start:start + args.batch]
        results = det(chunk, conf=args.min_conf, verbose=False, device=args.device)
        for j, res in enumerate(results):
            img_id = start + j
            h, w = res.orig_shape
            gts_by_img[img_id] = load_gt(frames[img_id], w, h)
            plist = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                for k in range(len(conf)):
                    plist.append((float(conf[k]), clamp(xyxy[k].tolist(), w, h)))
            preds_all[img_id] = plist
        if (start + args.batch) % 3200 < args.batch:
            el = time.time() - t0
            n = min(start + args.batch, len(frames))
            print(f"  {n}/{len(frames)} ({n/el:.0f} fps, {el:.0f}s)", flush=True)
    print(f"inference done in {time.time()-t0:.0f}s", flush=True)

    # ---- AP over the full score range ----
    flat = [(s, img, b) for img, lst in preds_all.items() for s, b in lst]
    flat.sort(key=lambda x: -x[0])
    thrs = np.arange(0.5, 1.0, 0.05)
    aps = [average_precision(flat, gts_by_img, t)[0] for t in thrs]
    ap50, rec50_all, prec50_all, ious50, npos = average_precision(flat, gts_by_img, 0.5)
    ap75 = average_precision(flat, gts_by_img, 0.75)[0]

    # ---- operating point (score >= op-conf): P / R / center / coverage ----
    op = {img: [(s, b) for s, b in lst if s >= args.op_conf] for img, lst in preds_all.items()}
    op_flat = [(s, img, b) for img, lst in op.items() for s, b in lst]
    op_flat.sort(key=lambda x: -x[0])
    _, rec50_op, prec50_op, ious50_op, _ = average_precision(op_flat, gts_by_img, 0.5)
    rec30_op = average_precision(op_flat, gts_by_img, 0.3)[1]
    cov, center = coverage_and_center(op, gts_by_img)

    report = {
        "n_frames": len(frames), "n_gt": npos,
        "n_pred_all": len(flat), "n_pred_op": len(op_flat),
        "op_conf": args.op_conf,
        "AP@[.50:.95]": float(np.mean(aps)), "AP50": float(ap50), "AP75": float(ap75),
        "precision@.50": prec50_op, "recall@.50": rec50_op, "recall@.30": float(rec30_op),
        "center_recall": center, "mean_matched_IoU@.50": float(np.mean(ious50_op)) if ious50_op else 0.0,
        "mean_GT_coverage_IoU": cov,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'metric':24s}{'WiLoR YOLO':>14s}")
    for k in ["AP@[.50:.95]", "AP50", "AP75", "precision@.50", "recall@.50",
              "recall@.30", "center_recall", "mean_matched_IoU@.50", "mean_GT_coverage_IoU"]:
        print(f"{k:24s}{report[k]:14.4f}")
    print(f"\nn_gt={npos}  preds(all>={args.min_conf})={len(flat)}  "
          f"preds(op>={args.op_conf})={len(op_flat)}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
