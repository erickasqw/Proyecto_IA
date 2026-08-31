"""Command-line entry point for training and evaluating the FER system."""

from dataset import get_dataloaders
from evaluate import evaluate_model
from model import EmotionResNet18
from train import train_model
from utils_zip import extract_datasets


def main():
    roots = extract_datasets()
    train_loader, validation_loader, test_loader = get_dataloaders(
        roots.values(), batch_size=24, num_workers=2)
    model = EmotionResNet18(freeze_backbone=True)
    checkpoint_path = "artifacts/best_emotion_model.pth"
    train_model(model, train_loader, validation_loader, epochs=12, patience=4,
                checkpoint_path=checkpoint_path,
                history_path="artifacts/training_history.json")
    evaluate_model(model, test_loader, checkpoint_path=checkpoint_path,
                   output_dir="artifacts/evaluation")


if __name__ == "__main__":
    main()