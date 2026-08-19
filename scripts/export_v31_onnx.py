import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FREQ = 513


class SepV31(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 192
        self.inp = nn.Conv1d(FREQ, hidden, 1)
        self.blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 8, 16, 32]:
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                "norm": nn.GroupNorm(12, hidden),
                "gate": nn.Conv1d(hidden, hidden, 1),
            }))
        self.out = nn.Conv1d(hidden, FREQ * 2, 1)

    def forward(self, x):
        h = F.silu(self.inp(x))
        for block in self.blocks:
            z = F.silu(block["norm"](block["conv"](h)))
            z = torch.sigmoid(block["gate"](z)) * z
            h = h + z
        logits = self.out(h).view(x.shape[0], 2, FREQ, x.shape[-1])
        return F.softmax(logits / 0.82, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SepV31()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, FREQ, 384)
    torch.onnx.export(
        model,
        dummy,
        output.as_posix(),
        opset_version=17,
        input_names=["spectrum"],
        output_names=["masks"],
        dynamic_axes={
            "spectrum": {2: "frames"},
            "masks": {3: "frames"},
        },
        dynamo=False,
    )

    print(f"Exported {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
