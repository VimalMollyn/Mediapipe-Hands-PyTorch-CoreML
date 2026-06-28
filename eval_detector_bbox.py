"""Evaluate the MediaPipe-Hands PyTorch detector's HAND BOUNDING BOXES on the
WHIM test set, three predicted-box representations side by side:

  palm : raw projected palm-detector box (detector stage only)
  roi  : the 2.6x square hand ROI the detector feeds to the crop (axis-aligned
         bounding box of the rotated rect)
  lm   : bounding box of the 21 predicted landmarks (detector -> landmark model)

GT = WHIM per-hand `bbox` [x1,y1,x2,y2] in pixels (tight full-hand box).
Metrics: COCO-style AP@[.50:.95], AP50, AP75, plus precision / recall / mean
matched-IoU @0.50 and mean GT-coverage IoU. Single "hand" class, multi-hand.

Run from the repo root with the uv env:
    uv run python eval_detector_bbox.py [--limit N] [--workers K] [--out eval.json]
"""
import argparse
import glob
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

TEST_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/test/anno"

_MODEL = None  # per-worker singleton


def _init_worker():
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from run_mediapipe_pytorch import HandLandmarkerTorch
    # num_hands high so we never cap multi-hand frames; we drive internals directly.
    # DETECTOR_PT lets us evaluate a fine-tuned detector graph (train_detector.py).
    det = os.environ.get("DETECTOR_PT", "models/hand_detector.pt")
    _MODEL = HandLandmarkerTorch(detector_path=det, num_hands=64, device="cpu")


def _clamp_box(b, w, h):
    x1, y1, x2, y2 = b
    x1 = min(max(x1, 0.0), w); x2 = min(max(x2, 0.0), w)
    y1 = min(max(y1, 0.0), h); y2 = min(max(y2, 0.0), h)
    return [x1, y1, x2, y2]


def _detect_boxes(model, img_rgb):
    """One detector pass -> (palm_preds, roi_preds, rects) in PIXELS.
    palm_preds/roi_preds: list of (score, [x1,y1,x2,y2]); rects: (score, rect)."""
    from fasthands.pipeline import (
        DETECT_SIZE, RECT_SCALE, RECT_SHIFT_Y, F, compute_rotation,
        crop_rotated_rect, decode_detections, letterbox_projection,
        project_detection, weighted_nms,
    )
    ih, iw = img_rgb.shape[:2]
    side = max(iw, ih)
    crop = crop_rotated_rect(img_rgb, F(0.5) * F(iw), F(0.5) * F(ih),
                             side, side, 0.0, DETECT_SIZE, cv2.BORDER_CONSTANT)
    rb, rs = model.detector(crop[None])
    dets = weighted_nms(decode_detections(rb[0], rs[0], model.anchors))
    project = letterbox_projection(iw, ih)

    palm, roi, rects = [], [], []
    for d in dets:
        pd = project_detection(d, project)
        score = float(pd["score"])
        # palm box (project_detection output is normalized)
        palm.append((score, [pd["xmin"] * iw, pd["ymin"] * ih,
                             (pd["xmin"] + pd["w"]) * iw, (pd["ymin"] + pd["h"]) * ih]))
        # DetectionsToRects + RectTransformation (scale 2.6, shift_y -0.5, square_long)
        cx = pd["xmin"] + pd["w"] / F(2.0)
        cy = pd["ymin"] + pd["h"] / F(2.0)
        w, h = pd["w"], pd["h"]
        rotation = compute_rotation(pd["kp"][0][0] * F(iw), pd["kp"][0][1] * F(ih),
                                    pd["kp"][2][0] * F(iw), pd["kp"][2][1] * F(ih))
        sin_a, cos_a = F(math.sin(rotation)), F(math.cos(rotation))
        if float(rotation) == 0.0:
            cx2, cy2 = cx, cy + h * RECT_SHIFT_Y
        else:
            x_shift = (-F(ih) * h * RECT_SHIFT_Y * sin_a) / F(iw)
            y_shift = (F(ih) * h * RECT_SHIFT_Y * cos_a) / F(ih)
            cx2, cy2 = cx + x_shift, cy + y_shift
        long_side = max(w * F(iw), h * F(ih))
        rw = long_side / F(iw) * RECT_SCALE
        rh = long_side / F(ih) * RECT_SCALE
        rect = (cx2, cy2, rw, rh, rotation)
        rects.append((score, rect))
        # axis-aligned bbox of the rotated 2.6x square (the crop region in pixels)
        pts = cv2.boxPoints(((float(cx2 * iw), float(cy2 * ih)),
                             (float(rw * iw), float(rh * ih)),
                             float(math.degrees(rotation))))
        roi.append((score, [float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max())]))
    return palm, roi, rects


def _process(npy_path):
    global _MODEL
    from fasthands.pipeline import deduplicate_hands
    jpg = npy_path[:-4] + ".jpg"
    bgr = cv2.imread(jpg)
    if bgr is None:
        return None
    ih, iw = bgr.shape[:2]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ground truth
    gt = []
    for hnd in np.load(npy_path, allow_pickle=True):
        b = _clamp_box([float(v) for v in np.asarray(hnd["bbox"], dtype=float)], iw, ih)
        if b[2] > b[0] and b[3] > b[1]:
            gt.append(b)

    try:
        palm, roi, rects = _detect_boxes(_MODEL, img_rgb)
        hands = []
        for score, rect in rects:
            hand = _MODEL._landmarks(img_rgb, *rect)
            if hand is not None:
                hand["_score"] = score
                hands.append(hand)
        hands = deduplicate_hands(hands, iw, ih)
        lm = []
        for hand in hands:
            p = hand["landmarks"][:, :2]
            lm.append((float(hand["_score"]),
                       [float(p[:, 0].min() * iw), float(p[:, 1].min() * ih),
                        float(p[:, 0].max() * iw), float(p[:, 1].max() * ih)]))
    except Exception as e:  # never let one frame kill the run
        return {"err": f"{os.path.basename(npy_path)}: {type(e).__name__}: {e}",
                "gt": gt, "palm": [], "roi": [], "lm": []}

    clamp = lambda lst: [(s, _clamp_box(b, iw, ih)) for s, b in lst]
    return {"gt": gt, "palm": clamp(palm), "roi": clamp(roi), "lm": clamp(lm)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = xb - xa, yb - ya
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(preds, gts_by_img, iou_thr):
    """preds: list of (score, img_id, box) sorted desc; gts_by_img: img->[box].
    COCO 101-point interpolation. Returns (ap, recall, precision, tp_ious)."""
    npos = sum(len(v) for v in gts_by_img.values())
    matched = {img: [False] * len(g) for img, g in gts_by_img.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds)); tp_ious = []
    for k, (score, img, box) in enumerate(preds):
        gts = gts_by_img.get(img, [])
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if matched[img][j]:
                continue
            i = iou_xyxy(box, gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[k] = 1; matched[img][best_j] = True; tp_ious.append(best_iou)
        else:
            fp[k] = 1
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / max(npos, 1)
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = prec[rec >= t]
        ap += (m.max() if m.size else 0.0) / 101
    final_rec = float(rec[-1]) if len(rec) else 0.0
    final_prec = float(prec[-1]) if len(prec) else 0.0
    return ap, final_rec, final_prec, tp_ious, npos


def evaluate_variant(results, key):
    preds, gts_by_img = [], {}
    for img_id, r in enumerate(results):
        gts_by_img[img_id] = r["gt"]
        for score, box in r[key]:
            preds.append((score, img_id, box))
    preds.sort(key=lambda x: -x[0])

    thrs = np.arange(0.5, 1.0, 0.05)
    aps = [average_precision(preds, gts_by_img, t)[0] for t in thrs]
    ap50, rec50, prec50, ious50, npos = average_precision(preds, gts_by_img, 0.5)
    ap75 = average_precision(preds, gts_by_img, 0.75)[0]
    rec30 = average_precision(preds, gts_by_img, 0.3)[1]
    rec10 = average_precision(preds, gts_by_img, 0.1)[1]

    # mean GT-coverage IoU + center-based detection recall (does ANY pred center
    # land inside a GT box) -- a box-size-independent "did it find the hand" signal
    by_img = {}
    for score, img_id, box in preds:
        by_img.setdefault(img_id, []).append(box)
    cov, center_hit, n_gt = [], 0, 0
    for img_id, r in enumerate(results):
        pl = by_img.get(img_id, [])
        for gb in r["gt"]:
            n_gt += 1
            cov.append(max((iou_xyxy(p, gb) for p in pl), default=0.0))
            for p in pl:
                cxp, cyp = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                if gb[0] <= cxp <= gb[2] and gb[1] <= cyp <= gb[3]:
                    center_hit += 1
                    break

    return {
        "AP@[.50:.95]": float(np.mean(aps)),
        "AP50": float(ap50), "AP75": float(ap75),
        "precision@.50": prec50, "recall@.50": rec50,
        "recall@.30": float(rec30), "recall@.10": float(rec10),
        "center_recall": center_hit / max(n_gt, 1),
        "mean_matched_IoU@.50": float(np.mean(ious50)) if ious50 else 0.0,
        "mean_GT_coverage_IoU": float(np.mean(cov)) if cov else 0.0,
        "n_pred": len(preds), "n_gt": npos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N frames (0=all)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="eval_detector_bbox.json")
    ap.add_argument("--detector", default=None,
                    help="path to a detector graph .pt (e.g. a fine-tuned "
                         "models/hand_detector_whim.pt); default = pretrained")
    args = ap.parse_args()
    if args.detector:
        os.environ["DETECTOR_PT"] = args.detector
        print(f"using detector: {args.detector}", flush=True)

    frames = sorted(glob.glob(os.path.join(TEST_ROOT, "*", "*.npy")))
    if args.limit:
        frames = frames[: args.limit]
    print(f"evaluating {len(frames)} frames on {args.workers} workers ...", flush=True)

    results, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for n, r in enumerate(ex.map(_process, frames, chunksize=16), 1):
            if r is None:
                continue
            if r.get("err"):
                errors.append(r["err"])
            results.append(r)
            if n % 2000 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(frames)}  ({n/el:.0f} fps, {el:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"processed {len(results)} frames in {el:.0f}s ({len(results)/el:.0f} fps), "
          f"errors={len(errors)}", flush=True)

    report = {"n_frames": len(results), "n_errors": len(errors),
              "variants": {k: evaluate_variant(results, k) for k in ("palm", "roi", "lm")}}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # pretty table
    cols = ["AP@[.50:.95]", "AP50", "AP75", "precision@.50", "recall@.50",
            "recall@.30", "recall@.10", "center_recall",
            "mean_matched_IoU@.50", "mean_GT_coverage_IoU"]
    names = {"palm": "palm box", "roi": "2.6x ROI", "lm": "landmark box"}
    print(f"\n{'metric':24s} " + "".join(f"{names[k]:>16s}" for k in ("palm", "roi", "lm")))
    for c in cols:
        print(f"{c:24s} " + "".join(f"{report['variants'][k][c]:16.4f}"
                                    for k in ("palm", "roi", "lm")))
    v = report["variants"]["lm"]
    print(f"\nn_gt={v['n_gt']}  preds: palm={report['variants']['palm']['n_pred']} "
          f"roi={report['variants']['roi']['n_pred']} lm={v['n_pred']}")
    print(f"report -> {args.out}")
    if errors:
        print(f"\nfirst errors: {errors[:5]}")


if __name__ == "__main__":
    main()
