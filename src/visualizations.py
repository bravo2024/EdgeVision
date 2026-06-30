"""visualizations.py - Edge vision benchmark visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#fafafa",
        "text.color": "#fafafa",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


def plot_model_comparison(benchmarks: Dict[str, Dict], val_accs: Dict[str, float] = None) -> plt.Figure:
    """Compare models on size, latency, accuracy."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    names = list(benchmarks.keys())
    sizes = [benchmarks[n]["model_size_mb"] for n in names]
    latencies = [benchmarks[n]["avg_inference_ms"] for n in names]
    params = [benchmarks[n]["param_count"] / 1e6 for n in names]

    colors = ["#22d3ee", "#a78bfa", "#f97316"]

    # Model size
    bars = axes[0].barh(names, sizes, color=colors[:len(names)], height=0.5)
    axes[0].set_xlabel("Size (MB)")
    axes[0].set_title("Model Size", fontsize=11, fontweight="bold")
    for bar, s in zip(bars, sizes):
        axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{s:.2f} MB", va="center", fontsize=9)

    # Latency
    bars = axes[1].barh(names, latencies, color=colors[:len(names)], height=0.5)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("Inference Latency", fontsize=11, fontweight="bold")
    for bar, l in zip(bars, latencies):
        axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{l:.1f} ms", va="center", fontsize=9)

    # Parameters
    bars = axes[2].barh(names, params, color=colors[:len(names)], height=0.5)
    axes[2].set_xlabel("Parameters (M)")
    axes[2].set_title("Parameter Count", fontsize=11, fontweight="bold")
    for bar, p in zip(bars, params):
        axes[2].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{p:.2f}M", va="center", fontsize=9)

    for ax in axes:
        ax.grid(axis="x", linestyle="--")
    fig.tight_layout()
    return fig


def plot_accuracy_vs_latency(benchmarks: Dict[str, Dict], val_accs: Dict[str, float]) -> plt.Figure:
    """Scatter plot of accuracy vs inference latency."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#22d3ee", "#a78bfa", "#f97316"]

    for i, (name, b) in enumerate(benchmarks.items()):
        acc = val_accs.get(name, 0) * 100
        ax.scatter(b["avg_inference_ms"], acc, s=200, color=colors[i % len(colors)],
                   edgecolor="#333", zorder=5, label=name)
        ax.annotate(name, (b["avg_inference_ms"], acc), textcoords="offset points",
                    xytext=(10, 5), fontsize=10, fontweight="bold")

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency Tradeoff", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    return fig


def plot_accuracy_vs_size(benchmarks: Dict[str, Dict], val_accs: Dict[str, float]) -> plt.Figure:
    """Scatter plot of accuracy vs model size."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#22d3ee", "#a78bfa", "#f97316"]

    for i, (name, b) in enumerate(benchmarks.items()):
        acc = val_accs.get(name, 0) * 100
        ax.scatter(b["model_size_mb"], acc, s=200, color=colors[i % len(colors)],
                   edgecolor="#333", zorder=5, label=name)
        ax.annotate(name, (b["model_size_mb"], acc), textcoords="offset points",
                    xytext=(10, 5), fontsize=10, fontweight="bold")

    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Model Size Tradeoff", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    return fig


def plot_training_curves(history: Dict) -> plt.Figure:
    """Loss and accuracy curves."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "o-", color="#22d3ee", linewidth=1.5, label="Train")
    ax1.plot(epochs, history["val_loss"], "o-", color="#f43f5e", linewidth=1.5, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--")

    ax2.plot(epochs, history["train_acc"], "o-", color="#22d3ee", linewidth=1.5, label="Train")
    ax2.plot(epochs, history["val_acc"], "o-", color="#f43f5e", linewidth=1.5, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--")

    fig.tight_layout()
    return fig


def plot_per_class_accuracy(per_class_acc: Dict) -> plt.Figure:
    """Bar chart of per-class accuracy."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(per_class_acc.keys())
    accs = list(per_class_acc.values())
    colors = ["#22c55e" if a > 0.9 else "#fbbf24" if a > 0.7 else "#f43f5e" for a in accs]

    bars = ax.bar(names, accs, color=colors, width=0.6, edgecolor="#333")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.15])
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_quantization_comparison(original_size: float, quantized_size: float,
                                  original_acc: float, quantized_acc: float) -> plt.Figure:
    """Bar chart comparing original vs quantized model."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    models = ["Original (FP32)", "Quantized (INT8)"]
    sizes = [original_size, quantized_size]
    accs = [original_acc, quantized_acc]
    colors = ["#a78bfa", "#22d3ee"]

    bars = ax1.bar(models, sizes, color=colors, width=0.5, edgecolor="#333")
    for bar, s in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{s:.2f} MB", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Size (MB)")
    ax1.set_title("Model Size", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--")

    bars = ax2.bar(models, accs, color=colors, width=0.5, edgecolor="#333")
    for bar, a in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{a:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.1])
    ax2.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--")

    fig.suptitle("Quantization Impact", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
