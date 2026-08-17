from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_carpet(depth, device):
    resolution = 3 ** depth
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
    print(f"depth: {depth}")
    print(f"resolution: {resolution}")
    print(f"retained pixels: {retained_pixels}")
    print(f"expected retained pixels: {expected_pixels}")

    if retained_pixels != expected_pixels:
        raise ValueError(f"depth {depth} retained {retained_pixels}, expected {expected_pixels}")

    carpet = mask.cpu().numpy()
    image = np.where(carpet, 0.0, 1.0)
    return image


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "outputs" / "sierpinski_depth_progression.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"selected device: {device}")

    depths = [1, 2, 3, 4, 5, 6]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for depth, axis in zip(depths, axes.flat):
        image = generate_carpet(depth, device)
        axis.imshow(image, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
        axis.set_title(f"Depth {depth}")
        axis.set_xticks([])
        axis.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"saved image: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
