"""Fine-tune the MediaPipe hand LANDMARK model on WHIM (Lightning + wandb).

Trains models/hand_landmarks_detector.pt to match WHIM GT: 2D landmarks (crop px),
world 3D landmarks (de-rotated GT root-relative, meters), and handedness. Presence
is held at 1 (all WHIM crops contain a hand). Loss = smooth-L1(xy) + smooth-L1(world,
in cm) + BCE(handedness) + BCE(presence). Reuses the detector trainer's weight
promotion / graph export / LR schedule.

  .venv-train/bin/python train_landmarks.py --train-subset 30000 --max-steps 400 --offline
  .venv-train/bin/python train_landmarks.py --wandb-project whim-hand-landmarks
"""
import argparse
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fasthands.pipeline import HAND_CONNECTIONS, LANDMARK_SIZE
from train_detector import export_graph, load_trainable_detector  # generic helpers
from whim_landmark_data import WHIMLandmarkDataModule, WHIMLandmarkDataset, list_frames, \
    hand_crop_and_targets, TEST_ROOT

N_VIZ = 8
EPS = 1e-6


class LandmarkLit(L.LightningModule):
    def __init__(self, model_path="models/hand_landmarks_detector.pt", lr=1e-4, wd=1e-4,
                 warmup_steps=1000, xy_w=1.0, world_w=1.0, hand_w=1.0, pres_w=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.net = load_trainable_detector(model_path)
        self._viz = None

    def forward(self, x):
        lm, pres, hand, world = self.net(x)
        n = x.shape[0]
        return (lm.reshape(n, 21, 3), pres.reshape(n), hand.reshape(n),
                world.reshape(n, 21, 3))

    def _step(self, batch, tag):
        crop, xy_t, world_t, side, valid = batch
        lm, pres, hand, world = self(crop)
        m = valid > 0.5
        if m.sum() == 0:
            return None
        pred_xy = lm[..., :2][m]
        xy_loss = F.smooth_l1_loss(pred_xy, xy_t[m])                       # crop px
        world_loss = F.smooth_l1_loss(world[m] * 100, world_t[m] * 100)   # cm
        # MSE (not BCE) on the probability outputs -- autocast/fp16-safe
        hand_loss = F.mse_loss(hand[m], side[m])
        pres_loss = F.mse_loss(pres[m], torch.ones_like(pres[m]))
        loss = (self.hparams.xy_w * xy_loss + self.hparams.world_w * world_loss
                + self.hparams.hand_w * hand_loss + self.hparams.pres_w * pres_loss)
        with torch.no_grad():
            px = (pred_xy - xy_t[m]).norm(dim=-1).mean()        # crop-px 2D error
        self.log_dict({f"{tag}/loss": loss, f"{tag}/xy": xy_loss, f"{tag}/world": world_loss,
                       f"{tag}/hand": hand_loss, f"{tag}/px_err": px},
                      prog_bar=(tag == "train"), batch_size=int(m.sum()),
                      on_step=(tag == "train"), on_epoch=(tag == "val"), sync_dist=True)
        return loss

    def training_step(self, b, _):
        return self._step(b, "train")

    def validation_step(self, b, _):
        self._step(b, "val")

    # ---- log GT (green) + predicted (red) landmark skeletons on val crops ----
    def _build_viz(self):
        frames = self.trainer.datamodule.val_set.frames
        idxs = np.linspace(0, len(frames) - 1, N_VIZ).astype(int)
        crops, xys = [], []
        for i in idxs:
            bgr = cv2.imread(frames[i][:-4] + ".jpg")
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hands = list(np.load(frames[i], allow_pickle=True))
            res = hand_crop_and_targets(rgb, hands[0]) if hands else None
            if res is None:
                continue
            crops.append(res[0]); xys.append(res[1])
        self._viz = (np.stack(crops), xys)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0 \
                or not isinstance(self.logger, WandbLogger):
            return
        if self._viz is None:
            self._build_viz()
        crops, xys = self._viz
        lm = self(torch.from_numpy(crops).to(self.device))[0].cpu().numpy()
        imgs = []
        for b in range(len(crops)):
            canvas = (crops[b] * 255).astype(np.uint8).copy()
            gt = xys[b]; pr = lm[b, :, :2]
            for a, c in HAND_CONNECTIONS:
                cv2.line(canvas, tuple(gt[a].astype(int)), tuple(gt[c].astype(int)), (0, 255, 0), 2)
                cv2.line(canvas, tuple(pr[a].astype(int)), tuple(pr[c].astype(int)), (255, 0, 0), 1)
            imgs.append(canvas)
        self.logger.log_image(key="val/landmarks", images=imgs, step=self.global_step)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        total = int(self.trainer.estimated_stepping_batches)
        warmup = min(self.hparams.warmup_steps, max(total - 1, 1))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            p = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

        return {"optimizer": opt, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), "interval": "step"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hand_landmarks_detector.pt")
    ap.add_argument("--export", default="models/hand_landmarks_detector_whim.pt")
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--val-subset", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--wandb-project", default="whim-hand-landmarks")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")

    dm = WHIMLandmarkDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                train_subset=args.train_subset, val_subset=args.val_subset)
    model = LandmarkLit(model_path=args.model, lr=args.lr, warmup_steps=args.warmup_steps)

    logger = WandbLogger(project=args.wandb_project, name=args.run_name,
                         offline=args.offline, log_model=False)
    logger.log_hyperparams(vars(args))
    ckpt_all = ModelCheckpoint(dirpath="checkpoints_lm", every_n_train_steps=args.ckpt_every,
                               save_top_k=-1, save_last=True, filename="lm-step{step}",
                               auto_insert_metric_name=False)
    ckpt_best = ModelCheckpoint(dirpath="checkpoints_lm", monitor="val/loss", mode="min",
                                save_top_k=3, filename="best-step{step}-{val/loss:.3f}",
                                auto_insert_metric_name=False)
    trainer = L.Trainer(accelerator="gpu", devices=[args.device], precision=args.precision,
                        max_steps=args.max_steps, max_epochs=args.max_epochs,
                        log_every_n_steps=10, logger=logger,
                        callbacks=[ckpt_all, ckpt_best, LearningRateMonitor(logging_interval="step")],
                        gradient_clip_val=10.0)
    trainer.fit(model, dm)

    export_graph(model.net, args.model, args.export)
    print(f"\nexported tuned landmark graph -> {args.export}", flush=True)


if __name__ == "__main__":
    main()
