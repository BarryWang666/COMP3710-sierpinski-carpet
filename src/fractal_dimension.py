from pathlib import Path
import math

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
    print(f"retained pixels: {retained_pixels}")
    print(f"expected pixels: {expected_pixels}")

    if retained_pixels != expected_pixels:
        raise ValueError(f"retained {retained_pixels}, expected {expected_pixels}")

    return mask


def count_occupied_boxes(mask, box_size):
    resolution = mask.shape[0]
    boxes_per_side = resolution // box_size
    blocks = mask.view(boxes_per_side, box_size, boxes_per_side, box_size)
    blocks = blocks.permute(0, 2, 1, 3)
    occupied = blocks.reshape(boxes_per_side, boxes_per_side, box_size * box_size).any(dim=2)
    return occupied.sum()


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "outputs" / "sierpinski_fractal_dimension.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"selected device: {device}")

    depth = 7
    resolution = 3 ** depth
    print(f"depth: {depth}")
    print(f"resolution: {resolution}")

    mask = generate_carpet(depth, device)
    box_sizes = [1, 3, 9, 27, 81, 243, 729]
    x_values = []
    y_values = []

    print("box_size epsilon occupied_boxes log(1/epsilon) log(N)")
    for box_size in box_sizes:
        occupied_boxes = count_occupied_boxes(mask, box_size)
        epsilon = torch.tensor(box_size / resolution, device=device, dtype=torch.float64)
        x_value = torch.log(1.0 / epsilon)
        y_value = torch.log(occupied_boxes.to(torch.float64))

        x_values.append(x_value)
        y_values.append(y_value)

        print(
            f"{box_size:8d} "
            f"{epsilon.item():.10f} "
            f"{occupied_boxes.item():14d} "
            f"{x_value.item():.10f} "
            f"{y_value.item():.10f}"
        )

    x_values = torch.stack(x_values).cpu().numpy()
    y_values = torch.stack(y_values).cpu().numpy()

    estimated_dimension, intercept = np.polyfit(x_values, y_values, 1)
    theoretical_dimension = math.log(8) / math.log(3)
    absolute_error = abs(estimated_dimension - theoretical_dimension)

    print(f"estimated dimension: {estimated_dimension:.10f}")
    print(f"theoretical dimension: {theoretical_dimension:.10f}")
    print(f"absolute error: {absolute_error:.10f}")

    fitted_y_values = estimated_dimension * x_values + intercept

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, label="Measured box counts")
    plt.plot(x_values, fitted_y_values, label=f"Fit: D = {estimated_dimension:.4f}")
    plt.xlabel("log(1 / epsilon)")
    plt.ylabel("log(N(epsilon))")
    plt.title("Sierpiński Carpet Box-Counting Dimension")
    plt.legend(title=f"Theoretical D = {theoretical_dimension:.4f}")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"saved image: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
