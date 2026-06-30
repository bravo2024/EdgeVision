"""tests/test_smoke.py - Smoke tests for EdgeVision pipeline."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_make_synthetic():
    from src.data import make_synthetic_cifar
    data = make_synthetic_cifar(n_per_class=5)
    assert data["train_images"].shape == (50, 3, 32, 32)
    assert data["train_labels"].shape == (50,)
    assert len(set(data["train_labels"])) == 10


def test_build_mobilenet():
    import torch
    from src.model import build_mobilenet_v2
    model = build_mobilenet_v2(10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    params = sum(p.numel() for p in model.parameters())
    assert params < 5_000_000  # < 5M params


def test_build_shufflenet():
    import torch
    from src.model import build_shufflenet_v2
    model = build_shufflenet_v2(10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    params = sum(p.numel() for p in model.parameters())
    assert params < 5_000_000  # < 5M params


def test_build_resnet18():
    import torch
    from src.model import build_resnet18
    model = build_resnet18(10, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_benchmark():
    import torch
    from src.model import build_mobilenet_v2, benchmark_model
    from torch.utils.data import DataLoader, TensorDataset

    model = build_mobilenet_v2(10)
    dummy_loader = DataLoader(
        TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, 10, (20,))),
        batch_size=5
    )
    bench = benchmark_model(model, dummy_loader, n_runs=5)
    assert "param_count" in bench
    assert "model_size_mb" in bench
    assert "avg_inference_ms" in bench
    assert bench["model_size_mb"] > 0


if __name__ == "__main__":
    test_make_synthetic()
    test_build_mobilenet()
    test_build_shufflenet()
    test_build_resnet18()
    test_benchmark()
    print("All smoke tests passed!")
