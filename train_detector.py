"""Continue-train the BlazePalm hand detector on WHIM (Lightning + wandb).

Fine-tunes models/hand_detector.pt with WHIM targets: box = full-hand GT bbox,
7 palm keypoints = WHIM joints [0,5,9,13,17,1,2] (see whim_data.py). Loss is an
SSD head loss over the 2016 point-anchors: focal loss for objectness + smooth-L1
for box and keypoint regression on positive anchors.

  uv-free run (uses the dedicated cu121 env):
    .venv-train/bin/python train_detector.py --train-subset 30000 --max-steps 800 \
        --offline                         # smoke verify
    .venv-train/bin/python train_detector.py                       # full run

After training, the tuned weights are written back into the graph format at
--export (default models/hand_detector_whim.pt) so run_mediapipe_pytorch.py /
eval_detector_bbox.py can load it directly.
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

from fasthands.pipeline import (DETECT_SIZE, decode_detections, generate_anchors,
                                weighted_nms)
from tflite_graph import TFLiteModule
from whim_data import WHIMDataModule, load_image_and_gt

N_VIZ = 8  # number of fixed val frames visualized in wandb each validation


def load_trainable_detector(path):
    """TFLiteModule with its float weight-buffers promoted to nn.Parameter."""
    net = TFLiteModule(path)
    for op in net.ops:                       # batch-agnostic heads (see tflite_graph)
        pass
    for name in list(net.weights.values()):
        b = getattr(net, name)
        if b.is_floating_point():
            del net._buffers[name]
            net.register_parameter(name, torch.nn.Parameter(b.clone()))
    return net


def export_graph(net, orig_path, out_path):
    """Write tuned weights back into the original graph dict so the pipeline can
    load them (inverts the PReLU-alpha NCHW permute done at TFLiteModule load)."""
    graph = torch.load(orig_path, weights_only=True)
    for k in list(graph["weights"].keys()):
        name = f"w{k}"
        if not hasattr(net, name):
            continue
        cur = getattr(net, name).detach().cpu()
        orig = graph["weights"][k]
        if cur.shape != orig.shape and cur.dim() == 3 and orig.dim() == 3:
            cur = cur.permute(1, 2, 0).contiguous()   # [C,1,1] -> [1,1,C] alpha
        graph["weights"][k] = cur
    torch.save(graph, out_path)


class DetectorLit(L.LightningModule):
    def __init__(self, detector_path="models/hand_detector.pt", lr=1e-4, wd=1e-4,
                 warmup_steps=1000, viz_every=2000, focal_alpha=0.25, focal_gamma=2.0,
                 box_w=1.0, kp_w=0.5, cls_w=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = load_trainable_detector(detector_path)
        self.anchors = generate_anchors()        # [2016,4] for decoding viz preds
        self._viz_val = None                     # fixed (inputs, gt_boxes) per split
        self._viz_train = None

    def forward(self, x):
        loc, score = self.net(x)             # [N,2016,18], [N,2016,1]
        return loc, score.squeeze(-1)        # [N,2016,18], [N,2016]

    def _focal(self, logits, target, valid):
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        a = self.hparams.focal_alpha
        alpha_t = a * target + (1 - a) * (1 - target)
        loss = alpha_t * (1 - pt).pow(self.hparams.focal_gamma) * ce
        return (loss * valid).sum()

    def _step(self, batch, tag):
        inp, loc_t, cls_t = batch
        loc_p, score_p = self(inp)
        pos = (cls_t == 1)
        valid = (cls_t != -1).float()
        npos = pos.sum().clamp(min=1).float()

        cls_loss = self._focal(score_p, (cls_t == 1).float(), valid) / npos
        if pos.any():
            box_loss = F.smooth_l1_loss(loc_p[..., :4][pos], loc_t[..., :4][pos],
                                        reduction="sum") / npos
            kp_loss = F.smooth_l1_loss(loc_p[..., 4:][pos], loc_t[..., 4:][pos],
                                       reduction="sum") / npos
        else:
            box_loss = kp_loss = torch.zeros((), device=inp.device)
        loss = (self.hparams.cls_w * cls_loss + self.hparams.box_w * box_loss
                + self.hparams.kp_w * kp_loss)

        bs = inp.shape[0]
        self.log_dict({f"{tag}/loss": loss, f"{tag}/cls": cls_loss,
                       f"{tag}/box": box_loss, f"{tag}/kp": kp_loss,
                       f"{tag}/pos_per_img": npos / bs},
                      prog_bar=(tag == "train"), batch_size=bs,
                      on_step=(tag == "train"), on_epoch=(tag == "val"), sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    # ---- log example images with GT (green) + predicted (red) boxes to wandb ----
    def _viz_set(self, frames):
        idxs = np.linspace(0, len(frames) - 1, min(N_VIZ, len(frames))).astype(int)
        imgs = [load_image_and_gt(frames[i]) for i in idxs]
        return np.stack([a for a, _ in imgs]), [b for _, b in imgs]

    @torch.no_grad()
    def _log_detections(self, key, viz):
        inputs, gts = viz
        loc, score = self(torch.from_numpy(inputs).to(self.device))
        S = DETECT_SIZE
        images, captions = [], []
        for b in range(len(inputs)):
            canvas = (inputs[b] * 255).astype(np.uint8).copy()  # RGB
            for g in gts[b]:                                    # GT green
                cv2.rectangle(canvas, (int(g[0] * S), int(g[1] * S)),
                              (int(g[2] * S), int(g[3] * S)), (0, 255, 0), 2)
            raw_b = loc[b].float().cpu().numpy()
            raw_s = score[b].float().cpu().numpy()[:, None]
            dets = sorted(weighted_nms(decode_detections(raw_b, raw_s, self.anchors)),
                          key=lambda d: -d["score"])[:10]
            for d in dets:                                      # predicted red
                x1, y1 = int(d["xmin"] * S), int(d["ymin"] * S)
                x2, y2 = int((d["xmin"] + d["w"]) * S), int((d["ymin"] + d["h"]) * S)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(canvas, f"{float(d['score']):.2f}", (x1, max(y1 - 2, 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            images.append(canvas)
            captions.append(f"gt={len(gts[b])} pred={len(dets)}")
        self.logger.log_image(key=key, images=images, caption=captions,
                              step=self.global_step)

    def _wandb_ready(self):
        return self.global_rank == 0 and isinstance(self.logger, WandbLogger)

    def on_train_start(self):
        """Baseline detections (val + train) before any optimizer step."""
        if not self._wandb_ready():
            return
        self._viz_val = self._viz_set(self.trainer.datamodule.val_set.frames)
        self._viz_train = self._viz_set(self.trainer.datamodule.train_set.frames)
        self._log_detections("val/detections", self._viz_val)
        self._log_detections("train/detections", self._viz_train)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._wandb_ready():
            return
        if self._viz_val is None:
            self._viz_val = self._viz_set(self.trainer.datamodule.val_set.frames)
        self._log_detections("val/detections", self._viz_val)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._wandb_ready() or self.global_step % self.hparams.viz_every != 0:
            return
        if self._viz_train is None:
            self._viz_train = self._viz_set(self.trainer.datamodule.train_set.frames)
        self._log_detections("train/detections", self._viz_train)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.wd)
        total = int(self.trainer.estimated_stepping_batches)
        warmup = min(self.hparams.warmup_steps, max(total - 1, 1))

        def lr_lambda(step):  # linear warmup -> cosine decay to 0 over all steps
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="models/hand_detector.pt")
    ap.add_argument("--export", default="models/hand_detector_whim.pt")
    ap.add_argument("--train-subset", type=int, default=0, help="0 = full train split")
    ap.add_argument("--val-subset", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="keep a checkpoint every N steps")
    ap.add_argument("--viz-every", type=int, default=2000, help="log train detections every N steps")
    ap.add_argument("--kp-w", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-epochs", type=int, default=5)
    ap.add_argument("--val-check-interval", type=float, default=1.0)
    ap.add_argument("--limit-val-batches", type=float, default=1.0)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--device", type=int, default=0, help="cuda device index")
    ap.add_argument("--wandb-project", default="whim-hand-detector")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--offline", action="store_true", help="wandb offline mode")
    args = ap.parse_args()

    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")

    dm = WHIMDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                        train_subset=args.train_subset, val_subset=args.val_subset)
    model = DetectorLit(detector_path=args.detector, lr=args.lr, kp_w=args.kp_w,
                        warmup_steps=args.warmup_steps, viz_every=args.viz_every)

    logger = WandbLogger(project=args.wandb_project, name=args.run_name,
                         offline=args.offline, log_model=False)
    logger.log_hyperparams(vars(args))

    # keep ALL checkpoints on a fixed step cadence (+ last), plus the best by val
    ckpt_all = ModelCheckpoint(dirpath="checkpoints", every_n_train_steps=args.ckpt_every,
                               save_top_k=-1, save_last=True, filename="det-step{step}",
                               auto_insert_metric_name=False)
    ckpt_best = ModelCheckpoint(dirpath="checkpoints", monitor="val/loss", mode="min",
                                save_top_k=3, filename="best-step{step}-{val/loss:.3f}",
                                auto_insert_metric_name=False)
    trainer = L.Trainer(
        accelerator="gpu", devices=[args.device], precision=args.precision,
        max_steps=args.max_steps, max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=10, logger=logger,
        callbacks=[ckpt_all, ckpt_best, LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=10.0,
    )
    trainer.fit(model, dm)

    export_graph(model.net, args.detector, args.export)
    print(f"\nexported tuned detector graph -> {args.export}", flush=True)
    print("evaluate it with:  uv run python eval_detector_bbox.py  "
          f"(point --detector at {args.export} via run_mediapipe_pytorch)", flush=True)


if __name__ == "__main__":
    main()
