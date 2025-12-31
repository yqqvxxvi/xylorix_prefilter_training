"""
Evaluate contrastive learning model using linear probe

This script trains a linear classifier on top of frozen encoder features
to evaluate the quality of learned representations.
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import WoodImageDataset
from src.data.transforms import get_val_transforms
from src.contrastive import SimCLRModel, LinearClassifier


def extract_features(model, dataloader, device):
    """
    Extract features using the frozen encoder

    Args:
        model: SimCLR model
        dataloader: DataLoader
        device: Device

    Returns:
        (features, labels) as numpy arrays
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)

            # Get encoder features (not projection)
            features = model.get_representation(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def train_linear_classifier(model, train_loader, val_loader, device, args):
    """
    Train linear classifier on top of frozen features

    Args:
        model: Pre-trained SimCLR model (encoder will be frozen)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device
        args: Arguments

    Returns:
        Trained classifier
    """
    # Freeze encoder
    for param in model.parameters():
        param.requires_grad = False

    model.eval()  # Set to eval mode

    # Create linear classifier
    classifier = LinearClassifier(
        input_dim=model.encoder_dim,
        num_classes=2  # Binary classification
    ).to(device)

    # Optimizer (only classifier parameters)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_classifier_state = None

    print("\nTraining linear classifier...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # Training
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            images, labels = images.to(device), labels.to(device)

            # Extract features (frozen encoder)
            with torch.no_grad():
                features = model.get_representation(images)

            # Forward pass through classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Extract features
                features = model.get_representation(images)

                # Forward pass
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                # Track metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Store for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier_state = classifier.state_dict().copy()
            print(f"  → New best validation accuracy: {val_acc:.2f}%")

    # Load best classifier
    classifier.load_state_dict(best_classifier_state)

    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return classifier


def evaluate_classifier(model, classifier, dataloader, device):
    """
    Evaluate trained classifier

    Args:
        model: SimCLR model (frozen)
        classifier: Trained linear classifier
        dataloader: Test dataloader
        device: Device

    Returns:
        Dictionary of metrics
    """
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)

            # Extract features
            features = model.get_representation(images)

            # Classify
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate contrastive learning with linear probe')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pre-trained SimCLR checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0'],
                        help='Encoder architecture')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale images')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing positive/ and negative/ subdirectories')
    parser.add_argument('--positive_dir', type=str, default='positive',
                        help='Name of positive class directory')
    parser.add_argument('--negative_dir', type=str, default='negative',
                        help='Name of negative class directory')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Train split ratio (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train linear classifier (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for linear classifier (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/linear_probe',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Linear Probe Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device)

    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model = SimCLRModel(
        encoder_name=args.encoder,
        pretrained=False,
        grayscale=args.grayscale
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load dataset
    print("\nLoading dataset...")
    data_root = Path(args.data_dir)
    positive_dir = data_root / args.positive_dir
    negative_dir = data_root / args.negative_dir

    # Use validation transforms (no augmentation)
    transform = get_val_transforms(grayscale=args.grayscale)

    full_dataset = WoodImageDataset.from_directories(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        transform=transform,
        grayscale=args.grayscale
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Train linear classifier
    classifier = train_linear_classifier(model, train_loader, val_loader, device, args)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_classifier(model, classifier, test_loader, device)

    print("\n" + "=" * 60)
    print("Test Set Results:")
    print("=" * 60)
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {test_metrics['f1']*100:.2f}%")
    print(f"AUC:       {test_metrics['auc']:.4f}")
    print("=" * 60)

    # Save results
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("Linear Probe Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write("\nTest Set Metrics:\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:    {test_metrics['recall']*100:.2f}%\n")
        f.write(f"F1 Score:  {test_metrics['f1']*100:.2f}%\n")
        f.write(f"AUC:       {test_metrics['auc']:.4f}\n")

    print(f"\nResults saved to {results_file}")

    # Save classifier
    classifier_path = output_dir / 'linear_classifier.pth'
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'test_metrics': test_metrics,
        'args': vars(args)
    }, classifier_path)
    print(f"Classifier saved to {classifier_path}")


if __name__ == '__main__':
    main()
