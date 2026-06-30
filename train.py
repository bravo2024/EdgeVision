"""train.py - EdgeVision training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="EdgeVision training pipeline")
    parser.add_argument("--model", choices=["mobilenet", "shufflenet", "resnet18"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.data import load_cifar10, make_torch_loaders
    from src.model import (
        build_mobilenet_v2, build_shufflenet_v2, build_resnet18,
        train_model, evaluate_model, benchmark_model
    )

    # Load data
    data = load_cifar10()
    print(f"Loaded CIFAR-10: {data['n_train']} train, {data['n_test']} test")

    # Build model
    if args.model == "mobilenet":
        model = build_mobilenet_v2(10)
    elif args.model == "shufflenet":
        model = build_shufflenet_v2(10)
    else:
        model = build_resnet18(10)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train
    train_loader, test_loader = make_torch_loaders(
        data["train_images"], data["train_labels"],
        data["test_images"], data["test_labels"]
    )

    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    result = train_model(model, train_loader, test_loader, num_epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate
    eval_result = evaluate_model(result["model"], test_loader, data["class_names"], device=device)
    print(f"Test Accuracy: {eval_result['accuracy']:.4f}")

    # Benchmark
    bench = benchmark_model(result["model"], test_loader, device=device)
    print(f"Model Size: {bench['model_size_mb']:.2f} MB")
    print(f"Latency: {bench['avg_inference_ms']:.1f} ms")

    # Save
    Path("models").mkdir(exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump({
            "accuracy": eval_result["accuracy"],
            "benchmark": bench,
        }, f, indent=2)
    print("\nSaved metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
