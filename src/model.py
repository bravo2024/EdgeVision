"""model.py - Lightweight CV models for edge deployment.

Models:
1. MobileNetV2: depthwise separable convolutions
2. ShuffleNetV2: channel shuffle operations
3. ResNet18: baseline for comparison

Mathematical foundations:
- Depthwise separable conv: standard conv = depthwise conv × pointwise conv
  Reduces parameters from K²·C_in·C_out to K²·C_in + C_in·C_out
- Channel shuffle: reshape → transpose → flatten for cross-group information flow
- Quantization: w_int8 = round(w / scale + zero_point), scale = (w_max - w_min) / 255
- ONNX export: graph representation for cross-platform inference
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time


def build_mobilenet_v2(num_classes: int = 10, pretrained: bool = False):
    """MobileNetV2 with inverted residuals.

    Architecture:
        Input (3, 32, 32)
        → Conv2d 3→16, 3×3, stride=1, padding=1 → BN → ReLU6
        → InvertedResidual block ×17 (expand→depthwise→project)
        → Conv2d 320→1280, 1×1 → BN → ReLU6
        → Global Average Pool
        → Dropout(0.2)
        → Linear 1280→num_classes

    Key innovation: Inverted residual block
        Input → 1×1 expand → 3×3 depthwise → 1×1 project (linear)
        Skip connection when stride=1 and input channels = output channels

    Math:
        Depthwise conv: (K×K×C_in) filters, each applied to one channel
        Pointwise conv: 1×1×C_in×C_out filter
        Total: K²·C_in + C_in·C_out vs K²·C_in·C_out for standard conv
    """
    import torch
    import torch.nn as nn

    def _make_div(v, divisor=8):
        return int(v + divisor / 2) // divisor * divisor

    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super().__init__()
            self.stride = stride
            assert stride in [1, 2]
            hidden_dim = int(inp * expand_ratio)
            self.use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ])
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            return self.conv(x)

    # MobileNetV2 configuration: (t, c, n, s)
    # t=expand_ratio, c=output_channels, n=num_blocks, s=stride
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    input_channel = 16
    last_channel = 1280

    layers = [nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
              nn.BatchNorm2d(input_channel),
              nn.ReLU6(inplace=True)]

    for t, c, n, s in cfg:
        output_channel = c
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(InvertedResidual(input_channel, output_channel, stride, t))
            input_channel = output_channel

    layers.extend([
        nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(last_channel),
        nn.ReLU6(inplace=True),
    ])

    features = nn.Sequential(*layers)
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(last_channel, num_classes),
    )

    class MobileNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.classifier = classifier

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return MobileNetV2()


def build_shufflenet_v2(num_classes: int = 10, pretrained: bool = False):
    """ShuffleNetV2 with channel shuffle.

    Architecture:
        Input (3, 32, 32)
        → Conv2d 3→24, 3×3, stride=1, padding=1 → BN → ReLU
        → ShuffleV2 blocks ×3 stages
        → Conv2d → BN → ReLU
        → Global Average Pool
        → Linear → num_classes

    Key innovation: Channel Shuffle
        After group conv, channels are split into groups.
        Channel shuffle ensures cross-group information flow:
        Reshape (batch, groups, channels_per_group, H, W)
        → Transpose (batch, channels_per_group, groups, H, W)
        → Flatten back to (batch, channels, H, W)
    """
    import torch
    import torch.nn as nn

    def channel_shuffle(x, groups):
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x

    class ShuffleV2Block(nn.Module):
        def __init__(self, inp, oup, stride):
            super().__init__()
            self.stride = stride
            self.pad = stride - 1
            branch_features = oup // 2

            if stride > 1:
                self.branch1 = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(branch_features),
                    nn.ReLU(inplace=True),
                )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            if self.stride > 1:
                x1 = self.branch1(x)
                x2 = self.branch2(x)
            else:
                x1, x2 = x.chunk(2, dim=1)
                x2 = self.branch2(x2)

            out = torch.cat([x1, x2], dim=1)
            out = channel_shuffle(out, 2)
            return out

    stages_repeats = [4, 8, 4]
    stages_out_channels = [24, 48, 96, 192]

    layers = [nn.Conv2d(3, stages_out_channels[0], 3, 1, 1, bias=False),
              nn.BatchNorm2d(stages_out_channels[0]),
              nn.ReLU(inplace=True)]

    input_channels = stages_out_channels[0]
    for repeats, output_channels in zip(stages_repeats, stages_out_channels[1:]):
        layers.append(ShuffleV2Block(input_channels, output_channels, 2))
        for _ in range(repeats - 1):
            layers.append(ShuffleV2Block(output_channels, output_channels, 1))
        input_channels = output_channels

    layers.extend([
        nn.Conv2d(input_channels, stages_out_channels[-1], 1, 1, 0, bias=False),
        nn.BatchNorm2d(stages_out_channels[-1]),
        nn.ReLU(inplace=True),
    ])

    features = nn.Sequential(*layers)
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(stages_out_channels[-1], num_classes),
    )

    class ShuffleNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.classifier = classifier

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return ShuffleNetV2()


def build_resnet18(num_classes: int = 10, pretrained: bool = False):
    """ResNet18 baseline for comparison."""
    import torch
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, num_epochs: int = 20,
                lr: float = 0.1, device: str = "cpu") -> Dict:
    """Train model with SGD + momentum + weight decay + cosine annealing.

    Optimizer (SGD with momentum):
        v_t = β·v_{t-1} + ∇L(θ)
        θ_t = θ_{t-1} - α·v_t

    Learning rate schedule: Cosine annealing
        α_t = α_min + 0.5·(α_max - α_min)·(1 + cos(π·t/T))
    """
    import torch
    import torch.nn as nn

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

    return {
        "model": model,
        "history": history,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
    }


def evaluate_model(model, test_loader, class_names: list, device: str = "cpu") -> Dict:
    """Evaluate model on test set."""
    import torch

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(np.mean(all_preds == all_labels))
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        per_class_acc[name] = float(np.mean(all_preds[mask] == i)) if mask.sum() > 0 else 0.0

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def benchmark_model(model, test_loader, device: str = "cpu", n_runs: int = 100) -> Dict:
    """Benchmark inference time, model size, and parameter count."""
    import torch

    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for p in model.parameters() for b in [p])
    model_size_mb = param_count * 4 / (1024 * 1024)  # float32

    sample_batch = next(iter(test_loader))
    images = sample_batch[0][:1].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(images)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(images)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    avg_time = np.mean(times)
    throughput = 1000.0 / avg_time

    return {
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "avg_inference_ms": float(avg_time),
        "throughput_img_per_sec": float(throughput),
        "n_runs": n_runs,
    }


def export_onnx(model, save_path: str, input_shape: Tuple = (1, 3, 32, 32)):
    """Export PyTorch model to ONNX format."""
    import torch

    model.eval()
    model.cpu()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return save_path


def quantize_model(model):
    """Post-training dynamic quantization (float32 → int8).

    Math:
        q = round(w / scale + zero_point)
        scale = (w_max - w_min) / 255
        zero_point = round(-w_min / scale)

    Reduces model size by ~4x with minimal accuracy loss.
    """
    import torch

    model_cpu = model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized
