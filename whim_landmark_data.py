"""WHIM -> MediaPipe hand-LANDMARK-model training data.

Each sample is ONE hand: the 224x224 ROI crop the pipeline would feed the landmark
model (built from the GT 21 joints via rect_from_landmarks), plus targets in the
model's own output space:

  xy    [21,2] : the 21 landmarks in crop pixels [0,224] (exact inverse of
                 pipeline.HandLandmarker._landmarks' projection)
  world [21,3] : root-relative GT joints_3d, de-rotated by the ROI angle so the
                 pipeline's WorldLandmarkProjection (R(+angle)) maps it back to
                 the GT camera-frame root-relative pose (meters)
  side   scalar: handedness target (1 right / 0 left)

GT 2D = project(joints_3d, K, trans). One random hand per frame per epoch.
"""
import glob
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L

from fasthands.pipeline import LANDMARK_SIZE, F, crop_rotated_rect, rect_from_landmarks

TRAIN_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/train/anno"
TEST_ROOT = "/media/vimal/T7_2TB/best_hand_detector/WHIM/test/anno"


def list_frames(root):
    return sorted(f for f in glob.glob(os.path.join(root, "*", "*.npy"))
                  if os.path.exists(f[:-4] + ".jpg"))


def _project(P3, K, trans):
    uv = (K @ (P3 + trans).T).T
    return uv[:, :2] / uv[:, 2:3]


def hand_crop_and_targets(rgb, hd, rng=None):
    """Build the 224 crop + (xy, world, side) targets for one GT hand.
    Returns None if the projection/ROI is degenerate."""
    ih, iw = rgb.shape[:2]
    J3 = np.asarray(hd["joints_3d"], np.float64)
    K = np.asarray(hd["K"], np.float64)
    tr = np.asarray(hd["trans"], np.float64)
    side = float(hd["side"])
    g2 = _project(J3, K, tr)
    if not np.isfinite(g2).all():
        return None
    lm_norm = np.concatenate([g2 / [iw, ih], np.zeros((21, 1))], 1).astype(np.float32)
    try:
        cx, cy, rw, rh, rot = rect_from_landmarks(lm_norm, iw, ih)
    except Exception:
        return None
    if rw <= 0 or rh <= 0:
        return None
    crop = crop_rotated_rect(rgb, F(cx) * F(iw), F(cy) * F(ih), F(rw) * F(iw),
                             F(rh) * F(ih), rot, LANDMARK_SIZE, cv2.BORDER_REPLICATE)
    c, s = math.cos(rot), math.sin(rot)
    nx = (g2[:, 0] / iw - cx) / rw
    ny = (g2[:, 1] / ih - cy) / rh
    x = c * nx + s * ny
    y = -s * nx + c * ny
    xy = np.stack([(x + 0.5) * LANDMARK_SIZE, (y + 0.5) * LANDMARK_SIZE], 1).astype(np.float32)
    G = J3 - J3[0]                                   # root-relative (wrist)
    wx = c * G[:, 0] + s * G[:, 1]
    wy = -s * G[:, 0] + c * G[:, 1]
    world = np.stack([wx, wy, G[:, 2]], 1).astype(np.float32)
    return crop, xy, world, np.float32(side)


class WHIMLandmarkDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def _empty(self):
        z = np.zeros((LANDMARK_SIZE, LANDMARK_SIZE, 3), np.float32)
        return (torch.from_numpy(z), torch.zeros(21, 2), torch.zeros(21, 3),
                torch.tensor(0.0), torch.tensor(0.0))   # last = valid flag

    def __getitem__(self, i):
        npy = self.frames[i]
        bgr = cv2.imread(npy[:-4] + ".jpg")
        if bgr is None:
            return self._empty()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = list(np.load(npy, allow_pickle=True))
        if not hands:
            return self._empty()
        hd = hands[np.random.randint(len(hands))]
        res = hand_crop_and_targets(rgb, hd)
        if res is None:
            return self._empty()
        crop, xy, world, side = res
        return (torch.from_numpy(crop), torch.from_numpy(xy), torch.from_numpy(world),
                torch.tensor(float(side)), torch.tensor(1.0))


class WHIMLandmarkDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8, train_subset=0, val_subset=2000,
                 seed=0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        rng = np.random.default_rng(self.hparams.seed)
        train = list_frames(TRAIN_ROOT)
        if self.hparams.train_subset:
            idx = rng.choice(len(train), min(self.hparams.train_subset, len(train)), replace=False)
            train = [train[i] for i in sorted(idx)]
        val = list_frames(TEST_ROOT)
        if self.hparams.val_subset and len(val) > self.hparams.val_subset:
            idx = rng.choice(len(val), self.hparams.val_subset, replace=False)
            val = [val[i] for i in sorted(idx)]
        self.train_set = WHIMLandmarkDataset(train)
        self.val_set = WHIMLandmarkDataset(val)
        print(f"[data] train={len(train)} val={len(val)} frames", flush=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=False,
                          drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=False,
                          persistent_workers=True)
