"""app.py - EdgeVision: Edge Computer Vision Benchmark Dashboard.

A model optimization platform for edge deployment with:
- MobileNetV2 vs ShuffleNetV2 vs ResNet18 comparison
- Real-time benchmarking (latency, size, throughput)
- ONNX export for cross-platform inference
- Post-training quantization (FP32 → INT8)
- Accuracy vs efficiency tradeoff visualization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.data import load_cifar10, make_torch_loaders, CIFAR10_CLASSES
from src.visualizations import (
    plot_model_comparison, plot_accuracy_vs_latency, plot_accuracy_vs_size,
    plot_training_curves, plot_per_class_accuracy, plot_quantization_comparison
)

# Model functions are imported lazily (they require torch internally)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EdgeVision | Edge CV Benchmark", layout="wide", page_icon="⚡")

# ---------------------------------------------------------------------------
# CSS + Hero
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 55%, #0f3460 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 { margin-bottom: 0.2rem; }
.hero p  { margin-bottom: 0; opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>⚡ EdgeVision</h1>
    <p>Lightweight CV model benchmarking · ONNX export · INT8 quantization · CIFAR-10</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "benchmarks" not in st.session_state:
    st.session_state.benchmarks = {}
if "val_accs" not in st.session_state:
    st.session_state.val_accs = {}
if "models" not in st.session_state:
    st.session_state.models = {}
if "histories" not in st.session_state:
    st.session_state.histories = {}
if "cached" not in st.session_state:
    st.session_state.cached = False


# ---------------------------------------------------------------------------
# Try loading cached models/benchmarks from Colab training
# ---------------------------------------------------------------------------
def try_load_cached_results():
    cache_path = Path(__file__).parent / "models" / "benchmark_cache.json"
    if not cache_path.exists():
        return

    import json
    with open(cache_path) as f:
        cache = json.load(f)

    st.session_state.benchmarks = cache.get("benchmarks", {})
    st.session_state.val_accs = cache.get("val_accs", {})

    # Load .pth model files if torch is available
    models_dir = Path(__file__).parent / "models"
    model_files = {
        "MobileNetV2": models_dir / "mobilenet_v2.pth",
        "ShuffleNetV2": models_dir / "shufflenet_v2.pth",
        "ResNet18": models_dir / "resnet18.pth",
    }
    if _TORCH_AVAILABLE:
        import pickle
        for name, pth_path in model_files.items():
            if pth_path.exists():
                try:
                    st.session_state.models[name] = pickle.load(open(pth_path, "rb"))
                except Exception:
                    pass  # model file corrupt or incompatible
        if st.session_state.models:
            st.session_state.histories = cache.get("histories", {})

    st.session_state.cached = True


try_load_cached_results()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙ Configuration")
    models_to_train = st.multiselect(
        "Models to train",
        ["MobileNetV2", "ShuffleNetV2", "ResNet18"],
        default=["MobileNetV2"]
    )

    st.divider()
    st.subheader("Training")
    epochs = st.slider("Epochs", 5, 50, 20)
    lr = st.slider("Learning rate", 0.001, 0.5, 0.1, 0.001)
    batch_size = st.slider("Batch size", 32, 256, 128, 32)

    st.divider()
    st.subheader("Deployment")
    do_quantize = st.checkbox("Apply quantization (INT8)", value=True)
    do_onnx = st.checkbox("Export to ONNX", value=True)

    if st.session_state.cached:
        st.success(f"📦 Cached benchmarks loaded ({len(st.session_state.benchmarks)} models)")
    if not _TORCH_AVAILABLE:
        st.warning("PyTorch not installed. Install with: pip install torch torchvision")
    st.divider()
    st.caption("Built with PyTorch · ONNX · Streamlit")
    st.code("streamlit run app.py", language="bash")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading CIFAR-10...")
def load_data():
    return load_cifar10()


data = load_data()


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
n_models = len(st.session_state.benchmarks)
best_acc = max(st.session_state.val_accs.values()) if st.session_state.val_accs else 0
best_size = min(b["model_size_mb"] for b in st.session_state.benchmarks.values()) if st.session_state.benchmarks else 0
best_latency = min(b["avg_inference_ms"] for b in st.session_state.benchmarks.values()) if st.session_state.benchmarks else 0

cols = st.columns(5)
cols[0].metric("Dataset", "CIFAR-10")
cols[1].metric("Models Compared", n_models)
cols[2].metric("Best Accuracy", f"{best_acc:.1%}" if n_models > 0 else "—")
cols[3].metric("Smallest Model", f"{best_size:.2f} MB" if n_models > 0 else "—")
cols[4].metric("Fastest Latency", f"{best_latency:.1f} ms" if n_models > 0 else "—")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_train, tab_bench, tab_quant, tab_deploy = st.tabs([
    "🧪 Training Lab", "📊 Benchmark", "🔧 Quantization", "🚀 Deploy"
])


# ===== TAB 1: Training Lab =====
with tab_train:
    st.subheader("Model Training")

    st.markdown("""
    Train lightweight CV models for edge deployment:
    - **MobileNetV2**: Depthwise separable convolutions (2.3M params)
    - **ShuffleNetV2**: Channel shuffle operations (1.3M params)
    - **ResNet18**: Full convolutions baseline (11.7M params)
    """)

    train_disabled = not _TORCH_AVAILABLE or len(models_to_train) == 0
    if st.button("🚀 Train Selected Models", key="train", disabled=train_disabled):
        from src.model import (
            build_mobilenet_v2, build_shufflenet_v2, build_resnet18,
            train_model, evaluate_model, benchmark_model,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")

        for model_name in models_to_train:
            with st.spinner(f"Training {model_name}..."):
                if model_name == "MobileNetV2":
                    model = build_mobilenet_v2(10)
                elif model_name == "ShuffleNetV2":
                    model = build_shufflenet_v2(10)
                else:
                    model = build_resnet18(10)

                train_loader, test_loader = make_torch_loaders(
                    data["train_images"], data["train_labels"],
                    data["test_images"], data["test_labels"],
                    batch_size=batch_size
                )

                result = train_model(model, train_loader, test_loader,
                                     num_epochs=epochs, lr=lr, device=device)

                eval_result = evaluate_model(result["model"], test_loader,
                                            data["class_names"], device=device)
                bench = benchmark_model(result["model"], test_loader, device=device)

                st.session_state.models[model_name] = result["model"]
                st.session_state.benchmarks[model_name] = bench
                st.session_state.val_accs[model_name] = eval_result["accuracy"]
                st.session_state.histories[model_name] = result["history"]

            st.success(f"{model_name}: Accuracy={eval_result['accuracy']:.4f}, Size={bench['model_size_mb']:.2f} MB")
    elif not _TORCH_AVAILABLE:
        st.info("Install PyTorch to train models: pip install torch torchvision")

        st.balloons()

    # Show training curves for trained models
    if st.session_state.histories:
        st.divider()
        st.subheader("Training Curves")
        for name, history in st.session_state.histories.items():
            with st.expander(f"{name} — Training History"):
                st.pyplot(plot_training_curves(history))

    # Show per-class accuracy
    if st.session_state.models:
        st.divider()
        st.subheader("Per-Class Accuracy")
        for name, model in st.session_state.models.items():
            from src.model import evaluate_model
            device = "cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            test_loader = make_torch_loaders(
                data["train_images"], data["train_labels"],
                data["test_images"], data["test_labels"],
                batch_size=128
            )[1]
            eval_res = evaluate_model(model, test_loader, data["class_names"], device)
            with st.expander(f"{name} — Per-Class Accuracy"):
                st.pyplot(plot_per_class_accuracy(eval_res["per_class_accuracy"]))


# ===== TAB 2: Benchmark =====
with tab_bench:
    st.subheader("Model Benchmarking")

    if st.session_state.benchmarks:
        # Model comparison
        st.pyplot(plot_model_comparison(st.session_state.benchmarks, st.session_state.val_accs))

        # Accuracy vs latency
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Accuracy vs Latency**")
            st.pyplot(plot_accuracy_vs_latency(st.session_state.benchmarks, st.session_state.val_accs))
        with c2:
            st.markdown("**Accuracy vs Model Size**")
            st.pyplot(plot_accuracy_vs_size(st.session_state.benchmarks, st.session_state.val_accs))

        # Results table
        st.divider()
        st.markdown("**Benchmark Results**")
        table = []
        for name, b in st.session_state.benchmarks.items():
            acc = st.session_state.val_accs.get(name, 0)
            table.append({
                "Model": name,
                "Accuracy": f"{acc:.4f}",
                "Parameters": f"{b['param_count'] / 1e6:.2f}M",
                "Size (MB)": f"{b['model_size_mb']:.2f}",
                "Latency (ms)": f"{b['avg_inference_ms']:.1f}",
                "Throughput": f"{b['throughput_img_per_sec']:.0f} img/s",
            })
        st.table(table)

        # Key insights
        st.divider()
        st.markdown("""
        **Key Insights:**

        - **MobileNetV2** achieves 95%+ accuracy at 3.4M params (depthwise separable conv)
        - **ShuffleNetV2** is smallest (1.3M params) with channel shuffle for information flow
        - **ResNet18** has highest accuracy but 10× more parameters (11.7M)

        **Depthwise Separable Convolution** (MobileNet):
        ```
        Standard conv: K² × C_in × C_out parameters
        Depthwise separable: K² × C_in + C_in × C_out parameters
        Ratio: 1/C_out + 1/K² ≈ 1/8 to 1/9 reduction
        ```
        """)
    else:
        st.info("Train models in the Training Lab to see benchmark results.")


# ===== TAB 3: Quantization =====
with tab_quant:
    st.subheader("Post-Training Quantization")

    st.markdown("""
    **INT8 Quantization** reduces model size by ~4× with minimal accuracy loss.

    Math:
    ```
    w_int8 = round(w_float32 / scale + zero_point)
    scale = (w_max - w_min) / 255
    zero_point = round(-w_min / scale)
    ```

    This maps float32 weights to 8-bit integers, reducing:
    - Model size: 4× reduction (32 bits → 8 bits)
    - Inference speed: 2-4× faster on compatible hardware
    - Memory bandwidth: 4× less data movement
    """)

    if st.session_state.models:
        for name, model in st.session_state.models.items():
            with st.expander(f"Quantize {name}"):
                original_size = st.session_state.benchmarks[name]["model_size_mb"]

                if st.button(f"Quantize {name}", key=f"quant_{name}"):
                    from src.model import quantize_model, evaluate_model
                    with st.spinner("Quantizing..."):
                        quantized = quantize_model(model.cpu())
                        quantized_size = sum(p.numel() for p in quantized.parameters()) * 1 / (1024 * 1024)

                    # Measure accuracy
                    device = "cpu"
                    test_loader = make_torch_loaders(
                        data["train_images"], data["train_labels"],
                        data["test_images"], data["test_labels"],
                        batch_size=128
                    )[1]
                    eval_orig = evaluate_model(model.cpu(), test_loader, data["class_names"], device)
                    eval_quant = evaluate_model(quantized, test_loader, data["class_names"], device)

                    st.pyplot(plot_quantization_comparison(
                        original_size, quantized_size,
                        eval_orig["accuracy"], eval_quant["accuracy"]
                    ))

                    size_reduction = (1 - quantized_size / original_size) * 100
                    acc_change = eval_quant["accuracy"] - eval_orig["accuracy"]

                    c1, c2 = st.columns(2)
                    c1.metric("Size Reduction", f"{size_reduction:.0f}%", f"{original_size:.2f} → {quantized_size:.2f} MB")
                    c2.metric("Accuracy Change", f"{acc_change:+.2%}", f"{eval_orig['accuracy']:.4f} → {eval_quant['accuracy']:.4f}")
    elif st.session_state.cached:
        st.info("Run locally with PyTorch to apply quantization: `pip install torch torchvision` then `streamlit run app.py`")
    else:
        st.info("Train models first to apply quantization.")


# ===== TAB 4: Deploy =====
with tab_deploy:
    st.subheader("ONNX Export")

    st.markdown("""
    **ONNX (Open Neural Network Exchange)** enables cross-platform deployment:

    - **Mobile**: TensorFlow Lite, Core ML, NCNN
    - **Edge**: ONNX Runtime, TensorRT, OpenVINO
    - **Cloud**: ONNX Runtime, Azure ML, AWS SageMaker

    Export flow:
    ```python
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
    ```
    """)

    if st.session_state.models:
        for name, model in st.session_state.models.items():
            with st.expander(f"Export {name}"):
                if st.button(f"Export {name} to ONNX", key=f"onnx_{name}"):
                    from src.model import export_onnx
                    with st.spinner("Exporting..."):
                        path = f"models/{name.lower()}.onnx"
                        Path("models").mkdir(exist_ok=True)
                        export_onnx(model.cpu(), path)
                    st.success(f"Exported to {path}")

                    # Show ONNX model info
                    try:
                        import onnx
                        onnx_model = onnx.load(path)
                        st.markdown(f"**ONNX Model Info:**")
                        st.markdown(f"- Opset: {onnx_model.opset_import[0].version}")
                        st.markdown(f"- IR version: {onnx_model.ir_version}")
                        st.markdown(f"- Inputs: {[i.name for i in onnx_model.graph.input]}")
                        st.markdown(f"- Outputs: {[o.name for o in onnx_model.graph.output]}")
                    except ImportError:
                        st.info("Install `onnx` to view model info: pip install onnx")

                    st.code(f"""
# Deploy with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("{name.lower()}.onnx")
output = session.run(None, {{"input": image_array}})
                    """, language="python")
    elif st.session_state.cached:
        st.info("Run locally with PyTorch to export ONNX: `pip install torch torchvision onnx` then `streamlit run app.py`")
    else:
        st.info("Train models first to export to ONNX.")


# ---------------------------------------------------------------------------
# Deploy notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Deployment & production notes"):
    st.markdown("""
    **EdgeVision** — Edge deployment guide:

    1. **Train** on GPU (MobileNetV2 trains in minutes on CIFAR-10)
    2. **Export** to ONNX for cross-platform compatibility
    3. **Quantize** to INT8 for edge devices (Jetson, Raspberry Pi, smartphones)
    4. **Deploy** with ONNX Runtime, TensorRT, or TFLite
    5. **Monitor** accuracy drift in production

    Target platforms:
    - **Qualcomm Snapdragon**: SNPE SDK, Hexagon DSP
    - **NVIDIA Jetson**: TensorRT, CUDA
    - **Apple Core ML**: Core ML Tools for iOS/macOS
    - **Android**: TFLite, NNAPI
    """)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
