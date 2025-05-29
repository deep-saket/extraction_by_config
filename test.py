import torch
import time


def benchmark_matmul(device, size=2048, iterations=50):
    # Check availability
    if device.type == 'mps' and not torch.backends.mps.is_available():
        print("→ MPS (Metal) backend not available; skipping.")
        return

    # Create random tensor
    x = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(10):
        _ = x @ x
    if device.type == 'mps':
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = x @ x
    if device.type == 'mps':
        torch.mps.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / iterations
    print(f"[{device.type.upper():>3}] Total: {total_time:.4f}s | "
          f"Avg per matmul: {avg_time:.4f}s")


if __name__ == "__main__":
    print("PyTorch versions:", torch.__version__)
    for dev in [torch.device('cpu'), torch.device('mps')]:
        benchmark_matmul(dev, size=4048, iterations=100)