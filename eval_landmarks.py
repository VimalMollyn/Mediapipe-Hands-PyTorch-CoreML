"""Evaluate the MediaPipe hand LANDMARK model (models/hand_landmarks_detector.pt)
on WHIM, in isolation from the detector.

For each GT hand we build the hand ROI the same way the pipeline's VIDEO-mode
tracking does -- but from the GROUND-TRUTH 21 joints (rect_from_landmarks) -- so
the landmark model is given an ideal crop and detection errors don't contaminate
the result. Then we compare its predicted 21 landmarks to WHIM GT:

  2D: per-joint pixel error + error normalized by the hand bbox diagonal + PCK
  3D: PA-MPJPE (Procrustes-aligned, mm) and root-relative MPJPE (mm) vs joints_3d
  presence: fraction of GT hands the model accepts (presence > 0.5)

GT 2D joints = project(joints_3d, K, trans) (validated to 0px vs stored verts).
WHIM joints_3d order matches MediaPipe's 21-joint order (verified earlier).

    uv run python eval_landmarks.py [--limit N] [--workers K]
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

TEST_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/test/anno"
PCK_ALPHAS = [0.05, 0.1, 0.2]   # fraction of bbox diagonal

_MODEL = None


def _init():
    global _MODEL
    import torch
    torch.set_num_threads(1)
    from run_mediapipe_pytorch import HandLandmarkerTorch
    _MODEL = HandLandmarkerTorch(num_hands=1, device="cpu")  # we drive _landmarks directly


def _project(P3, K, trans):
    uv = (K @ (P3 + trans).T).T
    return uv[:, :2] / uv[:, 2:3]


def _pa_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE (similarity: scale+rotation+translation).
    pred, gt: (21,3). Returns aligned-pred per-joint L2 (same units as gt)."""
    S1, S2 = pred.T, gt.T                       # (3,N)
    mu1, mu2 = S1.mean(1, keepdims=True), S2.mean(1, keepdims=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum()
    K = X1 @ X2.T
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(3)
    Z[-1, -1] = np.sign(np.linalg.det(V @ U.T))
    R = V @ Z @ U.T
    scale = np.trace(R @ K) / (var1 + 1e-12)
    S1_hat = scale * (R @ S1) + (mu2 - scale * (R @ mu1))
    return np.linalg.norm(S1_hat.T - gt, axis=1)


def _process(npy):
    global _MODEL
    from fasthands.pipeline import rect_from_landmarks
    bgr = cv2.imread(npy[:-4] + ".jpg")
    if bgr is None:
        return []
    ih, iw = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = []
    for hd in np.load(npy, allow_pickle=True):
        J3 = np.asarray(hd["joints_3d"], np.float64)            # (21,3) GT 3D
        K = np.asarray(hd["K"], np.float64)
        tr = np.asarray(hd["trans"], np.float64)
        gt2d = _project(J3, K, tr)                              # (21,2) px
        if not np.isfinite(gt2d).all():
            continue
        # ROI from GT landmarks (pipeline's HandLandmarksToRect), normalized
        lm_norm = np.concatenate([gt2d / [iw, ih], np.zeros((21, 1))], 1).astype(np.float32)
        try:
            rect = rect_from_landmarks(lm_norm, iw, ih)
            hand = _MODEL._landmarks(rgb, *rect)
        except Exception:
            hand = None
        rec = {"present": hand is not None}
        if hand is not None:
            pred2d = np.asarray(hand["landmarks"])[:, :2] * [iw, ih]     # px
            e2d = np.linalg.norm(pred2d - gt2d, axis=1)                  # (21,)
            x0, y0 = gt2d.min(0)
            x1, y1 = gt2d.max(0)
            ref = float(np.hypot(x1 - x0, y1 - y0)) + 1e-6               # bbox diag px
            pred_w = np.asarray(hand["world_landmarks"], np.float64)     # (21,3) m
            pa = _pa_mpjpe(pred_w, J3)                                   # per-joint (m)
            rr = np.linalg.norm((pred_w - pred_w[0]) - (J3 - J3[0]), axis=1)
            rec.update(e2d=e2d.tolist(), ref=ref,
                       pa_mm=float(pa.mean() * 1000), rr_mm=float(rr.mean() * 1000))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="eval_landmarks.json")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(TEST_ROOT, "*", "*.npy")))
    if args.limit:
        frames = frames[: args.limit]
    print(f"landmark eval on {len(frames)} frames, {args.workers} workers ...", flush=True)

    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for n, r in enumerate(ex.map(_process, frames, chunksize=16), 1):
            recs.extend(r)
            if n % 4000 == 0:
                print(f"  {n}/{len(frames)} ({n/(time.time()-t0):.0f} fps)", flush=True)
    el = time.time() - t0

    n_gt = len(recs)
    present = [r for r in recs if r["present"]]
    pres_rate = len(present) / max(n_gt, 1)
    # flatten per-joint 2D errors over accepted hands
    e2d = np.array([e for r in present for e in r["e2d"]])               # (21*Np,)
    refs = np.array([r["ref"] for r in present for _ in range(21)])
    norm = e2d / refs
    pck = {f"PCK@{a}": float((e2d < a * refs).mean()) for a in PCK_ALPHAS}
    pa = np.array([r["pa_mm"] for r in present])
    rr = np.array([r["rr_mm"] for r in present])

    report = {
        "n_frames": len(frames), "n_gt_hands": n_gt, "n_present": len(present),
        "presence_rate": pres_rate,
        "mean_2D_px": float(e2d.mean()), "median_2D_px": float(np.median(e2d)),
        "mean_2D_norm_bbox": float(norm.mean()), **pck,
        "PA_MPJPE_mm": float(pa.mean()), "MPJPE_rootrel_mm": float(rr.mean()),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nprocessed {n_gt} GT hands in {el:.0f}s ({len(frames)/el:.0f} fps)")
    print(f"{'presence_rate':22s}{pres_rate:10.4f}")
    for k in ["mean_2D_px", "median_2D_px", "mean_2D_norm_bbox", *pck,
              "PA_MPJPE_mm", "MPJPE_rootrel_mm"]:
        print(f"{k:22s}{report[k]:10.4f}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
