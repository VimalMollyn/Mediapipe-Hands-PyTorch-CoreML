"""WHIM -> BlazePalm detector training data.

Each frame becomes a 192x192 letterboxed detector input plus dense per-anchor
targets in the detector's own output space:

  loc target [2016, 18] = [cx, cy, w, h, (kx,ky) x 7]  in "192 units"
      (the same units the model regresses: normalized-letterbox coord * 192,
       relative to the anchor centre -- the exact inverse of
       fasthands.pipeline.decode_detections)
  cls target [2016]      = 1 positive / 0 negative / -1 ignore

Box target = WHIM's full-hand `bbox` (re-targeted from palm -> whole hand).
The 7 palm keypoints are WHIM joints [0,5,9,13,17,1,2] = wrist, index/middle/
ring/pinky MCP, thumb CMC, thumb MCP (verified empirically against the model's
own keypoint outputs). Anchors are point anchors (w=h=1), so matching is
"anchor centre inside the GT box" (FCOS-style); anchors in a dilated band around
a box are ignored to keep the negative set clean.
"""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from fasthands.pipeline import DETECT_SIZE, F, crop_rotated_rect, generate_anchors

# WHIM 21-joint indices for the 7 BlazePalm palm keypoints (see module docstring)
KP_JOINTS = [0, 5, 9, 13, 17, 1, 2]
N_KP = 7
IGNORE_DILATE = 1.3   # anchors whose centre is in [box, box*IGNORE_DILATE) -> ignore

TRAIN_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/train/anno"
TEST_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/test/anno"


def list_frames(root, with_image=True):
    """All annotated frames under root that also have an extracted .jpg."""
    out = []
    for npy in glob.glob(os.path.join(root, "*", "*.npy")):
        if not with_image or os.path.exists(npy[:-4] + ".jpg"):
            out.append(npy)
    return sorted(out)


def _project(P3, K, trans):
    uv = (K @ (P3 + trans).T).T
    return uv[:, :2] / uv[:, 2:3]


def _dilate(b, f):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class WHIMDetectorDataset(Dataset):
    def __init__(self, frames, anchors):
        self.frames = frames
        self.anchors = anchors.astype(np.float32)        # [2016,4] cx,cy,w,h in [0,1]
        self.acx = self.anchors[:, 0]
        self.acy = self.anchors[:, 1]

    def __len__(self):
        return len(self.frames)

    def _targets(self, boxes, kps):
        """boxes [M,4] xyxy, kps [M,7,2] all in letterbox-normalized [0,1]."""
        A = len(self.anchors)
        loc = np.zeros((A, 4 + 2 * N_KP), np.float32)
        cls = np.zeros((A,), np.float32)           # 0 = negative
        if len(boxes) == 0:
            return loc, cls
        acx, acy = self.acx, self.acy
        # assign each anchor to the smallest GT box whose (dilated) region holds it
        best_area = np.full((A,), np.inf, np.float32)
        for m, b in enumerate(boxes):
            inside = (acx >= b[0]) & (acx <= b[2]) & (acy >= b[1]) & (acy <= b[3])
            db = _dilate(b, IGNORE_DILATE)
            band = (acx >= db[0]) & (acx <= db[2]) & (acy >= db[1]) & (acy <= db[3])
            cls[band & (cls == 0)] = -1          # ignore band (don't override pos)
            area = (b[2] - b[0]) * (b[3] - b[1])
            take = inside & (area < best_area)
            if not take.any():
                continue
            best_area[take] = area
            cls[take] = 1
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            w, h = b[2] - b[0], b[3] - b[1]
            loc[take, 0] = (cx - acx[take]) * DETECT_SIZE
            loc[take, 1] = (cy - acy[take]) * DETECT_SIZE
            loc[take, 2] = w * DETECT_SIZE
            loc[take, 3] = h * DETECT_SIZE
            for k in range(N_KP):
                loc[take, 4 + 2 * k] = (kps[m, k, 0] - acx[take]) * DETECT_SIZE
                loc[take, 5 + 2 * k] = (kps[m, k, 1] - acy[take]) * DETECT_SIZE
        return loc, cls

    def __getitem__(self, i):
        npy = self.frames[i]
        bgr = cv2.imread(npy[:-4] + ".jpg")
        if bgr is None:
            # return an all-negative sample rather than crash the loader
            inp = np.zeros((DETECT_SIZE, DETECT_SIZE, 3), np.float32)
            loc, cls = self._targets(np.zeros((0, 4)), np.zeros((0, N_KP, 2)))
            return torch.from_numpy(inp), torch.from_numpy(loc), torch.from_numpy(cls)
        ih, iw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        side = max(iw, ih)
        inp = crop_rotated_rect(rgb, F(0.5) * F(iw), F(0.5) * F(ih), side, side,
                                0.0, DETECT_SIZE, cv2.BORDER_CONSTANT)  # [192,192,3] in [0,1]

        def to_lb(px, py):  # pixel -> letterbox-normalized [0,1]
            return (px - iw / 2) / side + 0.5, (py - ih / 2) / side + 0.5

        boxes, kps = [], []
        for h in np.load(npy, allow_pickle=True):
            b = np.asarray(h["bbox"], dtype=np.float64)
            x1, y1 = to_lb(b[0], b[1])
            x2, y2 = to_lb(b[2], b[3])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                continue
            J = _project(np.asarray(h["joints_3d"], np.float64),
                         np.asarray(h["K"], np.float64), np.asarray(h["trans"], np.float64))
            kp = np.array([to_lb(J[j, 0], J[j, 1]) for j in KP_JOINTS], np.float32)
            boxes.append([x1, y1, x2, y2])
            kps.append(kp)
        boxes = np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32)
        kps = np.array(kps, np.float32) if kps else np.zeros((0, N_KP, 2), np.float32)
        loc, cls = self._targets(boxes, kps)
        return (torch.from_numpy(inp), torch.from_numpy(loc), torch.from_numpy(cls))


class WHIMDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=6, train_subset=0, val_subset=2000,
                 seed=0):
        super().__init__()
        self.save_hyperparameters()
        self.anchors = generate_anchors()

    def setup(self, stage=None):
        rng = np.random.default_rng(self.hparams.seed)
        train = list_frames(TRAIN_ROOT)
        if self.hparams.train_subset:
            idx = rng.choice(len(train), min(self.hparams.train_subset, len(train)),
                             replace=False)
            train = [train[i] for i in sorted(idx)]
        val = list_frames(TEST_ROOT)
        if self.hparams.val_subset and len(val) > self.hparams.val_subset:
            idx = rng.choice(len(val), self.hparams.val_subset, replace=False)
            val = [val[i] for i in sorted(idx)]
        self.train_set = WHIMDetectorDataset(train, self.anchors)
        self.val_set = WHIMDetectorDataset(val, self.anchors)
        print(f"[data] train={len(train)} val={len(val)} frames", flush=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers,
                          pin_memory=True, persistent_workers=True)
