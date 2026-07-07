# EdgeVision

Benchmarking lightweight computer-vision models for edge deployment.

## Objective

Picking a CV model for an edge device is a three-way trade-off between accuracy, latency, and model size — and the right answer changes with the hardware budget. EdgeVision makes that trade-off measurable instead of guessed: it trains MobileNetV2, ShuffleNetV2, and ResNet18 on CIFAR-10 under identical conditions, benchmarks each on latency / size / throughput, then walks the two standard compression steps (ONNX export, post-training INT8 quantization) and shows what each one costs in accuracy.

Concretely, the project answers:

1. How much accuracy do you give up moving from ResNet18 (11.7M params) to MobileNetV2 (2.3M) or ShuffleNetV2 (1.3M)?
2. What does INT8 dynamic quantization buy in size, and what does it cost in accuracy, per architecture?
3. What do the exported ONNX graphs look like for runtime deployment (TensorRT, Core ML, TFLite, ONNX Runtime)?

## Results

With 20 epochs of SGD on CIFAR-10 (longer training shifts all three up, gaps stay similar):

| Model | Params | Test accuracy |
|---|---|---|
| ResNet18 | 11.7M | ~85% |
| MobileNetV2 | 2.3M | ~82% |
| ShuffleNetV2 | 1.3M | ~80% |

The headline: a 5× parameter cut (ResNet18 → MobileNetV2) costs roughly 3 accuracy points at this budget. Latency and throughput are measured on whatever device runs the benchmark, so absolute numbers vary by machine — the dashboard reports them per run rather than this README claiming fixed values.

## Running it

```bash
pip install -r requirements.txt
pip install torch torchvision        # needed for training/benchmarking
python train.py --model mobilenet    # downloads CIFAR-10 (~170 MB) on first run
pytest -q
streamlit run app.py
```

`train.py` takes `--model {mobilenet,shufflenet,resnet18}`, `--epochs`, `--lr`, and `--synthetic` (random data, for a fast pipeline check), and writes weights plus `models/metrics.json`.

Torch is intentionally left out of `requirements.txt` so the dashboard can be hosted on Streamlit Cloud, where the PyTorch wheel exceeds the resource budget. Without torch the app runs in demo mode: the UI, tabs, and explanations all load, and training/quantization/export unlock once torch is installed. A `EdgeVision_Colab_Training.ipynb` notebook is included for training on a free Colab GPU and bringing the results back to the dashboard.

## Dashboard

- **Training Lab** — train any subset of the three models with configurable epochs / LR / batch size; per-epoch curves and per-class accuracy
- **Benchmark** — latency (ms), size (MB), throughput (img/s) side by side, plus accuracy-vs-latency and accuracy-vs-size scatter plots
- **Quantization** — dynamic INT8 quantization with before/after size and accuracy deltas
- **Deploy** — ONNX export with graph metadata (opset, inputs/outputs) and an ONNX Runtime inference snippet

## Repository layout

```
src/
  data.py            CIFAR-10 loading via torchvision, synthetic fallback, augmented loaders
  model.py           the three architectures + train/evaluate/benchmark/quantize/export
  visualizations.py  matplotlib figures used by the dashboard
train.py             CLI training pipeline
app.py               Streamlit dashboard
tests/               smoke tests (model shapes, param budgets, benchmark keys)
```

Data: CIFAR-10 (60,000 32×32 RGB images, 10 classes), fetched automatically by torchvision. License: MIT.
