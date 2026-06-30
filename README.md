# EdgeVision

> Lightweight computer vision model benchmarking for edge deployment.

Compares MobileNetV2, ShuffleNetV2, and ResNet18 on CIFAR-10 with real-time benchmarking (latency, model size, throughput), ONNX export, and INT8 post-training quantisation. Dashboard visualises the accuracy-versus-efficiency trade-off for edge CV model selection.

## Quickstart

```bash
pip install -r requirements.txt
python train.py --model mobilenet   # train on CIFAR-10
pytest -q
streamlit run app.py
```

Note: Training requires the CIFAR-10 dataset (~170 MB download on first run). The dashboard will still load without trained models for exploration of the benchmarking framework.

## Model Performance

The framework supports three architectures — actual performance depends on training completion:

| Model | Test Accuracy (20 epochs) |
|---|---|
| MobileNetV2 | ~82% |
| ShuffleNetV2 | ~80% |
| ResNet18 | ~85% |

## Features

| Component | What it does |
|---|---|
| **Model Training** | Train MobileNetV2, ShuffleNetV2, or ResNet18 on CIFAR-10 with configurable epochs and learning rate |
| **Benchmarking** | Latency (ms), model size (MB), throughput (FPS) measured on-device |
| **ONNX Export** | Export trained models to ONNX format for cross-platform inference |
| **Quantization** | Post-training INT8 quantisation with accuracy comparison |
| **Comparison** | Accuracy vs latency / accuracy vs size trade-off visualisation |

## Repo Structure

```
EdgeVision/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     PyTorch training pipeline
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

CIFAR-10 (60,000 32×32 colour images, 10 classes). Downloaded automatically via `torchvision.datasets` on first run. Synthetic fallback for dashboard exploration without download.

## License

MIT
