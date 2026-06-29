"""Offline tool: convert the extracted graph specs (.pt) to CoreML .mlpackage
via torch.jit.trace + coremltools.

Usage:
    python tflite_to_coreml.py [--fp32]

FP16 (default) is required for the Neural Engine; FP32 runs on CPU/GPU only
but matches the PyTorch outputs more closely.
"""

import argparse

import coremltools as ct
import numpy as np
import torch

from tflite_graph import TFLiteModule


class TupleWrapper(torch.nn.Module):
    """coremltools wants a tuple return, not a list."""

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return tuple(self.mod(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", action="store_true",
                        help="convert at float32 (CPU/GPU only, closer to pytorch)")
    parser.add_argument("--detector", default="models/hand_detector.pt",
                        help="detector graph .pt (e.g. a fine-tuned *_whim.pt)")
    parser.add_argument("--landmark", default="models/hand_landmarks_detector.pt",
                        help="landmark graph .pt (e.g. a fine-tuned *_whim.pt)")
    args = parser.parse_args()

    precision = ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16
    suffix = "_fp32" if args.fp32 else ""

    # .mlpackage is named after the input .pt (so *_whim.pt -> *_whim.mlpackage)
    models = [(args.detector, (1, 192, 192, 3)), (args.landmark, (1, 224, 224, 3))]
    for pt_path, shape in models:
        out_tmpl = pt_path[:-3] + "{suffix}.mlpackage"
        module = TFLiteModule(pt_path).eval()
        output_names = [module.names[i] for i in module.output_ids]
        example = torch.rand(shape)
        with torch.no_grad():
            traced = torch.jit.trace(TupleWrapper(module), example, strict=False)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=shape, dtype=np.float32)],
            outputs=[ct.TensorType(name=n) for n in output_names],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS13,
        )
        out_path = out_tmpl.format(suffix=suffix)
        mlmodel.save(out_path)

        # sanity check against the torch module
        ref = [t.numpy() for t in module(example)]
        pred = ct.models.MLModel(out_path).predict({"image": example.numpy()})
        for name, r in zip(output_names, ref):
            d = np.abs(pred[name] - r).max()
            print(f"{out_path} {name:12s} max_abs_diff={d:.3e}")


if __name__ == "__main__":
    main()
