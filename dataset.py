"""Dataset discovery, label normalization, transforms, and DataLoaders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms


CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
LABEL_TO_INDEX = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
LABEL_ALIASES = {
    "anger": "angry", "angry": "angry", "disgust": "disgust",
    "fear": "fear", "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad", "surprise": "surprise",
    "neutral": "neutral",
}
RAF_LABELS = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
              "5": "sad", "6": "angry", "7": "neutral"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _label_for_path(path: Path, source: str) -> int | None:
    parts = [part.lower() for part in path.parts]
    # RAF-DB uses 1..7 directory names; only apply this mapping to RAF.
    if source == "raf":
        for part in reversed(parts[:-1]):
            if part in RAF_LABELS:
                return LABEL_TO_INDEX[RAF_LABELS[part]]
    for part in reversed(parts[:-1]):
        normalized = LABEL_ALIASES.get(part)
        if normalized is not None:
            return LABEL_TO_INDEX[normalized]
    return None


class FacialEmotionDataset(Dataset):
    """A single dataset view over FER2013, CK+, and RAF-DB images."""

    def __init__(self, data_roots: Iterable[str | Path], transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for root_value in data_roots:
            root = Path(root_value)
            source = root.name.lower()
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {root}")
            for image_path in sorted(root.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = _label_for_path(image_path, source)
                if label is not None:
                    self.samples.append((image_path, label))
        if not self.samples:
            raise RuntimeError("No labeled images were found in the dataset roots.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except OSError as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(), transforms.RandomRotation(10),
        transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return train, evaluation


def get_dataloaders(data_roots: Iterable[str | Path], batch_size: int = 32,
                    num_workers: int = 2, seed: int = 42):
    """Return train, validation, and test loaders using an 80/10/10 split."""
    train_transform, evaluation_transform = _transforms()
    all_data = FacialEmotionDataset(data_roots)
    sizes = [int(len(all_data) * 0.8), int(len(all_data) * 0.1)]
    sizes.append(len(all_data) - sum(sizes))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_data), generator=generator).tolist()
    train_indices = indices[:sizes[0]]
    validation_indices = indices[sizes[0]:sizes[0] + sizes[1]]
    test_indices = indices[sizes[0] + sizes[1]:]
    train_data = FacialEmotionDataset(data_roots, transform=train_transform)
    evaluation_data = FacialEmotionDataset(data_roots, transform=evaluation_transform)
    train_set = Subset(train_data, train_indices)
    validation_set = Subset(evaluation_data, validation_indices)
    test_set = Subset(evaluation_data, test_indices)
    loader_args = {"batch_size": batch_size, "num_workers": num_workers,
                   "pin_memory": torch.cuda.is_available(),
                   "persistent_workers": num_workers > 0}
    train_labels = torch.tensor([train_data.samples[index][1] for index in train_indices])
    class_counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    sample_weights = class_counts[train_labels].clamp_min(1).rsqrt()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    return (DataLoader(train_set, sampler=sampler, **loader_args),
            DataLoader(validation_set, shuffle=False, **loader_args),
            DataLoader(test_set, shuffle=False, **loader_args))