from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "outputs" / "sierpinski_baseline.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"selected device: {device}")

    depth = 7
    resolution = 3 ** depth
    print(f"depth: {depth}")
    print(f"resolution: {resolution}")

    coords = torch.arange(resolution, device=device, dtype=torch.int64)
    x = coords.clone()
    y = coords.clone()
    mask = torch.ones((resolution, resolution), device=device, dtype=torch.bool)

    for _ in range(depth):
        x_middle = (x % 3) == 1
        y_middle = (y % 3) == 1
        centre_third = y_middle[:, None] & x_middle[None, :]
        mask = mask & ~centre_third
        x = x // 3
        y = y // 3

    retained_pixels = mask.sum().item()
    expected_pixels = 8 ** depth
    print(f"retained pixels: {retained_pixels}")
    print(f"expected pixels: {expected_pixels}")

    carpet = mask.cpu().numpy()
    image = np.where(carpet, 0.0, 1.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"saved image: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
