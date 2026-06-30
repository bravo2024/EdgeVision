"""data.py - CIFAR-10 data loading for edge vision benchmarking.

CIFAR-10 is the standard benchmark for lightweight CV models:
- 60,000 images (32×32 RGB), 10 classes
- 50,000 train / 10,000 test
- Perfect for benchmarking MobileNet, ShuffleNet, etc.

Mathematical foundations:
- Data augmentation: random crop, horizontal flip, normalize
- Normalization: x' = (x - μ) / σ with ImageNet statistics
"""

from pathlib import Path

import numpy as np
from typing import Dict, Optional, Tuple

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
N_CLASSES = 10
IMG_SIZE = 32
CHANNELS = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_cifar10(data_dir: Optional[str] = None) -> Dict:
    """Load CIFAR-10 via torchvision.

    Returns dict with:
        train_images, train_labels: training set
        test_images, test_labels: test set
        class_names: list of 10 class names
        n_train, n_test: dataset sizes
    """
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, TensorDataset

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        if data_dir is None:
            import tempfile
            data_dir = str(Path(__file__).parent.parent / "data" / "raw")

        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # Convert to numpy for consistency
        train_images = np.stack([train_set[i][0].numpy() for i in range(len(train_set))])
        train_labels = np.array([train_set[i][1] for i in range(len(train_set))])
        test_images = np.stack([test_set[i][0].numpy() for i in range(len(test_set))])
        test_labels = np.array([test_set[i][1] for i in range(len(test_set))])

        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "class_names": CIFAR10_CLASSES,
            "n_train": len(train_labels),
            "n_test": len(test_labels),
            "img_size": IMG_SIZE,
            "channels": CHANNELS,
        }
    except ImportError:
        return make_synthetic_cifar()


def make_synthetic_cifar(n_per_class: int = 50, seed: int = 42) -> Dict:
    """Generate synthetic CIFAR-like data for demo."""
    rng = np.random.default_rng(seed)
    n = n_per_class * N_CLASSES

    train_images = rng.uniform(0, 1, (n, CHANNELS, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    train_labels = np.repeat(np.arange(N_CLASSES), n_per_class)

    test_images = rng.uniform(0, 1, (n, CHANNELS, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    test_labels = np.repeat(np.arange(N_CLASSES), n_per_class)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "class_names": CIFAR10_CLASSES,
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "img_size": IMG_SIZE,
        "channels": CHANNELS,
    }


def make_torch_loaders(train_images: np.ndarray, train_labels: np.ndarray,
                       test_images: np.ndarray, test_labels: np.ndarray,
                       batch_size: int = 128, augment: bool = True):
    """Convert numpy arrays to PyTorch DataLoaders."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    test_transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def apply_transform(images, transform):
        transformed = []
        for img in images:
            img_tensor = torch.FloatTensor(img)
            if transform is not None:
                img_tensor = transform(img_tensor)
            transformed.append(img_tensor)
        return torch.stack(transformed)

    train_imgs = apply_transform(train_images, train_transform)
    test_imgs = apply_transform(test_images, test_transform)

    train_loader = DataLoader(
        TensorDataset(train_imgs, torch.LongTensor(train_labels)),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(test_imgs, torch.LongTensor(test_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader
