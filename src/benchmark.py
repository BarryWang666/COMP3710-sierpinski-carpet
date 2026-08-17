from pathlib import Path
import time

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

    return mask


def verify_mask(mask, depth):
    retained_pixels = mask.sum().item()
    expected_pixels = 8 ** depth
    if retained_pixels != expected_pixels:
        raise ValueError(f"depth {depth} retained {retained_pixels}, expected {expected_pixels}")


def benchmark_depth(depth, device, repetitions):
    warmup_mask = generate_carpet(depth, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    verify_mask(warmup_mask, depth)

    times = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        mask = generate_carpet(depth, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time
        verify_mask(mask, depth)
        times.append(elapsed_time)

    return sum(times) / len(times)


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "outputs" / "sierpinski_cpu_gpu_benchmark.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available; running CPU benchmark only.")

    depths = [4, 5, 6, 7, 8]
    resolutions = [3 ** depth for depth in depths]
    repetitions = 3

    cpu_device = torch.device("cpu")
    cpu_times = []
    gpu_times = []

    for depth in depths:
        cpu_times.append(benchmark_depth(depth, cpu_device, repetitions))

        if cuda_available:
            gpu_device = torch.device("cuda")
            gpu_times.append(benchmark_depth(depth, gpu_device, repetitions))

    if cuda_available:
        print("depth resolution CPU_mean_seconds GPU_mean_seconds speedup")
        for depth, resolution, cpu_time, gpu_time in zip(depths, resolutions, cpu_times, gpu_times):
            speedup = cpu_time / gpu_time
            print(f"{depth:5d} {resolution:10d} {cpu_time:16.6f} {gpu_time:16.6f} {speedup:8.3f}")
    else:
        print("depth resolution CPU_mean_seconds")
        for depth, resolution, cpu_time in zip(depths, resolutions, cpu_times):
            print(f"{depth:5d} {resolution:10d} {cpu_time:16.6f}")

    resolution_values = np.array(resolutions)
    cpu_time_values = np.array(cpu_times)

    plt.figure(figsize=(8, 6))
    plt.plot(resolution_values, cpu_time_values, marker="o", label="CPU")

    if cuda_available:
        gpu_time_values = np.array(gpu_times)
        plt.plot(resolution_values, gpu_time_values, marker="o", label="GPU")

    plt.xlabel("Resolution")
    plt.ylabel("Execution time (seconds)")
    plt.title("Sierpiński Carpet PyTorch CPU vs GPU Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"saved image: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
