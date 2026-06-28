# Continue-training the hand detector on WHIM

Fine-tunes the BlazePalm hand **detector** (`models/hand_detector.pt`) on the
[WHIM](https://huggingface.co/datasets/rolpotamias/WHIM) dataset, and evaluates
hand bounding boxes against WHIM's ground truth. PyTorch Lightning + Weights &
Biases.

## Data layout (expected)
```
/media/vimal/T7_2TB/best_hand_detector/WHIM/{train,test}/anno/<video_id>/<frame>.{npy,jpg}
```
Each `.npy` is a WHIM per-hand annotation list (`bbox`, `joints_3d`, `K`,
`trans`, ...); `.jpg` is the extracted frame. Paths are set at the top of
`whim_data.py` (`TRAIN_ROOT` / `TEST_ROOT`).

## Environment
The repo's default env ships torch `cu130` (too new for this box's CUDA-12.4
driver), so training uses a dedicated venv:
```sh
uv venv .venv-train --python 3.11
uv pip install --python .venv-train "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-train lightning wandb opencv-python-headless "numpy<2"
uv pip install --python .venv-train --no-deps -e .     # make `fasthands` importable
```

## What it trains
Targets are built in the detector's own output space (the inverse of
`fasthands.pipeline.decode_detections`), over the 2016 point-anchors:
- **box** = WHIM full-hand `bbox` (re-targeted from palm → whole hand)
- **7 palm keypoints** = WHIM joints `[0,5,9,13,17,1,2]` (wrist, index/middle/
  ring/pinky MCP, thumb CMC, thumb MCP — verified against the model's own
  keypoint outputs)
- **matching**: anchor centre inside the GT box (point anchors, w=h=1); a
  dilated band around each box is ignored
- **loss**: focal objectness + smooth-L1 box/keypoint regression on positives

## Run
```sh
# smoke verify (subset, a few hundred steps, wandb offline)
.venv-train/bin/python train_detector.py --train-subset 30000 --max-steps 300 --offline

# full run (train split -> val on test split); log online (after `wandb login`)
.venv-train/bin/python train_detector.py --wandb-project whim-hand-detector
```
Checkpoints land in `checkpoints/`. After fitting, the tuned weights are written
back into the graph format at `--export` (default
`models/hand_detector_whim.pt`) so the existing pipeline can load them.

## Evaluate (boxes)
`eval_detector_bbox.py` scores three predicted-box variants (raw palm box, 2.6×
hand ROI, and the 21-landmark box) vs WHIM's full-hand GT with COCO-style
AP / precision / recall / coverage. Runs on CPU (the detector is tiny):
```sh
uv run python eval_detector_bbox.py                                   # pretrained baseline
uv run python eval_detector_bbox.py --detector models/hand_detector_whim.pt   # fine-tuned
```
Note the GT box is the tight 778-vertex mesh box, so strict IoU is dominated by
box-definition; read `center_recall` / `recall@.10` for "did it find the hand"
and `AP50` for box-fit. Baseline numbers are in `eval_detector_bbox.json`.
