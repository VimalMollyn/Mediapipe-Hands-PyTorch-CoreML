"""Pure-PyTorch executor for graph specs exported by tflite_to_torch.py.

Activations are kept in NCHW (PyTorch-native) layout throughout; the input is
permuted once from TFLite's NHWC and ops with layout-dependent semantics
(RESHAPE, CONCATENATION, MEAN, PAD) are remapped at load time. Conv inputs are
therefore contiguous NCHW tensors, which is both the fast path and numerically
identical (layout does not change values or accumulation order).
"""

import torch
import torch.nn.functional as F


def _same_pad(in_size: int, k: int, stride: int, dilation: int) -> tuple[int, int]:
    """TFLite SAME padding: total pad split with the extra pixel at the end."""
    out_size = (in_size + stride - 1) // stride
    eff_k = (k - 1) * dilation + 1
    pad = max((out_size - 1) * stride + eff_k - in_size, 0)
    return pad // 2, pad - pad // 2


def _activate(x: torch.Tensor, act: str) -> torch.Tensor:
    if act == "none":
        return x
    if act == "relu":
        return F.relu(x)
    if act == "relu6":
        return torch.clamp(x, 0.0, 6.0)
    if act == "tanh":
        return torch.tanh(x)
    raise NotImplementedError(act)


# NHWC dim index -> NCHW dim index, for 4D tensors
_NHWC_TO_NCHW_AXIS = {0: 0, 1: 2, 2: 3, 3: 1}


class TFLiteModule(torch.nn.Module):
    def __init__(self, spec_path: str):
        super().__init__()
        graph = torch.load(spec_path, weights_only=True)
        self.ops = graph["ops"]
        self.input_ids = graph["inputs"]
        self.output_ids = graph["outputs"]
        self.names = graph["names"]
        shapes = graph["shapes"]
        self.weights = {}
        for k, w in graph["weights"].items():
            name = f"w{k}"
            self.register_buffer(name, w)
            self.weights[k] = name

        # --- remap layout-dependent ops from NHWC to NCHW semantics, and
        # resolve all static shape-dependent values (paddings, groups) so the
        # forward pass makes no .shape queries (keeps jit traces clean) ---
        for op in self.ops:
            t, ins, outs, o = op["type"], op["inputs"], op["outputs"], op["options"]
            if t in ("CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D"):
                in_h, in_w = shapes[ins[0]][1], shapes[ins[0]][2]  # NHWC
                if t == "MAX_POOL_2D":
                    kh, kw = o["filter"]
                    dh = dw = 1
                else:
                    kh, kw = graph["weights"][ins[1]].shape[2:]
                    dh, dw = o["dilation"]
                    o["groups"] = 1 if t == "CONV_2D" else shapes[ins[0]][3]
                o["conv_pad"], o["pre_pad"] = (0, 0), None
                if o["padding"] == "same":
                    pt, pb = _same_pad(in_h, kh, o["stride"][0], dh)
                    pl, pr = _same_pad(in_w, kw, o["stride"][1], dw)
                    if pt == pb and pl == pr and t != "MAX_POOL_2D":
                        o["conv_pad"] = (pt, pl)  # symmetric: conv2d's own padding
                    elif pt or pb or pl or pr:
                        o["pre_pad"] = (pl, pr, pt, pb)
            if t == "PRELU" and len(shapes[ins[1]]) == 3:
                # alpha [1, 1, C] -> [C, 1, 1] so it broadcasts over NCHW
                w = getattr(self, self.weights[ins[1]])
                setattr(self, self.weights[ins[1]],
                        w.permute(2, 0, 1).contiguous())
            elif t == "CONCATENATION" and len(shapes[outs[0]]) == 4:
                o["axis"] = _NHWC_TO_NCHW_AXIS[o["axis"] % 4]
            elif t == "MEAN" and len(shapes[ins[0]]) == 4:
                o["axes"] = [_NHWC_TO_NCHW_AXIS[a % 4] for a in o["axes"]]
            elif t == "PAD" and len(shapes[ins[0]]) == 4:
                n, h, w_, c = o["paddings"]
                o["paddings"] = [n, c, h, w_]  # NCHW row order
            elif t == "RESHAPE":
                # reshape semantics are defined on the NHWC tensor: flag a
                # permute-back when the input is 4D (rare: detector heads only)
                o["from_4d"] = len(shapes[ins[0]]) == 4
                o["to_4d"] = len(o["shape"]) == 4

    def _const(self, idx: int) -> torch.Tensor:
        return getattr(self, self.weights[idx])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: [1, H, W, C] float32 (NHWC, like TFLite). Returns the model's
        output tensors in model order (4D outputs would be NHWC)."""
        env: dict[int, torch.Tensor] = {
            self.input_ids[0]: x.permute(0, 3, 1, 2).contiguous()
        }

        def get(idx: int) -> torch.Tensor:
            return env[idx] if idx in env else self._const(idx)

        for op in self.ops:
            t = op["type"]
            ins, outs, o = op["inputs"], op["outputs"], op["options"]

            if t in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                xin = get(ins[0])
                w = self._const(ins[1])
                b = self._const(ins[2]) if len(ins) > 2 else None
                if o["pre_pad"] is not None:
                    xin = F.pad(xin, o["pre_pad"])
                y = F.conv2d(xin, w, b, stride=o["stride"], padding=o["conv_pad"],
                             dilation=o["dilation"], groups=o["groups"])
                env[outs[0]] = _activate(y, o["activation"])

            elif t == "PRELU":
                xin = get(ins[0])
                alpha = self._const(ins[1])
                env[outs[0]] = torch.where(xin >= 0, xin, xin * alpha)

            elif t == "ADD":
                env[outs[0]] = _activate(get(ins[0]) + get(ins[1]), o["activation"])

            elif t == "MAX_POOL_2D":
                xin = get(ins[0])
                if o["pre_pad"] is not None:
                    xin = F.pad(xin, o["pre_pad"], value=float("-inf"))
                y = F.max_pool2d(xin, o["filter"], o["stride"])
                env[outs[0]] = _activate(y, o["activation"])

            elif t == "PAD":
                p = o["paddings"]
                flat = []
                for dim in reversed(p):
                    flat.extend(dim)
                env[outs[0]] = F.pad(get(ins[0]), flat)

            elif t == "RESIZE_BILINEAR":
                env[outs[0]] = F.interpolate(
                    get(ins[0]), size=tuple(o["size"]), mode="bilinear",
                    align_corners=o["align_corners"],
                )

            elif t == "RESHAPE":
                xin = get(ins[0])
                if o["from_4d"]:
                    xin = xin.permute(0, 2, 3, 1)  # back to NHWC semantics
                # The exported shapes bake in batch=1; use the runtime batch size
                # for dim 0 so the graph also runs batched (training). Identical
                # for batch=1 (xin.shape[0] == 1), enables N>1 otherwise.
                shape = list(o["shape"])
                shape[0] = xin.shape[0]
                y = xin.reshape(shape)
                if o["to_4d"]:
                    y = y.permute(0, 3, 1, 2)
                env[outs[0]] = y

            elif t == "CONCATENATION":
                y = torch.cat([get(i) for i in ins], dim=o["axis"])
                env[outs[0]] = _activate(y, o["activation"])

            elif t == "MEAN":
                env[outs[0]] = get(ins[0]).mean(dim=o["axes"], keepdim=o["keep_dims"])

            elif t == "FULLY_CONNECTED":
                w = self._const(ins[1])
                b = self._const(ins[2]) if len(ins) > 2 else None
                env[outs[0]] = _activate(F.linear(get(ins[0]), w, b), o["activation"])

            elif t == "LOGISTIC":
                env[outs[0]] = torch.sigmoid(get(ins[0]))

            else:
                raise NotImplementedError(t)

        outs = []
        for i in self.output_ids:
            y = env[i]
            if y.dim() == 4:
                y = y.permute(0, 2, 3, 1)
            outs.append(y)
        return outs
