"""
Visualise convolutional activation maps for a trained MNIST CNN.

Usage (after training with `mnist_cnn.py`):

    python visualize_activations.py --checkpoint mnist_cnn.pt --sample-index 0 \
                                    --output-dir activations

The script will:
1. Rebuild the same CNN architecture.
2. Load the saved `.pt` checkpoint.
3. Run a single test image through the network while forward-hooks capture
   activations of `conv1`, `conv2`, `conv3`.
4. Save tiled PNGs for each layer (e.g. `conv1_maps.png`) in the given directory.

If `--show` is passed, the figures are also displayed on-screen.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

from model import SmallCNN


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_activation_hooks(
    model: nn.Module, layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    """Attach forward hooks to the given `layer_names` (attributes of `model`).

    Returns a dict that will be populated with activations at runtime.
    """
    activations: Dict[str, torch.Tensor] = {}

    def save_activation(name):
        def hook(_, __, output):
            activations[name] = output.detach().cpu()

        return hook

    for name in layer_names:
        layer = getattr(model, name)
        layer.register_forward_hook(save_activation(name))
    return activations


def tile_and_save(maps: torch.Tensor, title: str, out_path: Path, show: bool = False):
    """Tile the CxHxW activation maps into a grid and save as PNG."""
    # maps: (C, H, W)
    c, h, w = maps.shape
    cols = int(math.ceil(math.sqrt(c)))
    rows = int(math.ceil(c / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx in range(c):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(maps[idx], cmap="gray")
        plt.axis("off")
        plt.title(f"ch {idx}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise CNN activation maps on MNIST"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="mnist_cnn.pt",
        help="Path to trained .pt file",
    )
    parser.add_argument(
        "--sample-index", type=int, default=0, help="Index of test image to visualise"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("activations"),
        help="Where to save PNG files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively as well as saving",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # rebuild model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    # select one sample from the MNIST test set
    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    img, label = test_ds[args.sample_index]
    img_batch = img.unsqueeze(0).to(device)  # shape (1, 1, 28, 28)

    # register hooks to capture activations
    activations = register_activation_hooks(model, ["conv1", "conv2", "conv3"])

    with torch.no_grad():
        _ = model(img_batch)

    # plot and save activation maps
    for name, act in activations.items():
        # act shape: (1, C, H, W) -> (C, H, W)
        tile_and_save(
            act.squeeze(0),
            f"{name} activation maps",
            args.output_dir / f"{name}_maps_{args.sample_index}.png",
            show=args.show,
        )

    print(f"Activation maps saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
