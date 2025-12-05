"""
Training utilities for CNN models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict

from ..utils.metrics import compute_metrics
import pandas as pd


class CNNTrainer:
    """Trainer for CNN models"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 save_dir: Path = Path('models')):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device (cuda/mps/cpu)
            scheduler: Optional learning rate scheduler
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Handle different output formats
            if outputs.shape[-1] == 1:
                # Binary classification with single output (BCEWithLogitsLoss)
                # Criterion expects logits, so we pass raw outputs
                outputs = outputs.squeeze()  # (batch_size, 1) -> (batch_size,)
                loss = self.criterion(outputs, labels.float())
                # For predictions, apply sigmoid to get probabilities
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:
                # Multi-class or binary with 2 outputs (CrossEntropyLoss)
                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

        return {'loss': epoch_loss, **metrics}

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='[Val]'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                # Handle different output formats
                if outputs.shape[-1] == 1:
                    # Binary classification with single output (BCEWithLogitsLoss)
                    outputs = outputs.squeeze()  # (batch_size, 1) -> (batch_size,)
                    loss = self.criterion(outputs, labels.float())
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                else:
                    # Multi-class or binary with 2 outputs (CrossEntropyLoss)
                    loss = self.criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)[:, 1]

                running_loss += loss.item() * images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        return {'loss': epoch_loss, **metrics}

    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Train model

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch, num_epochs)
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics.get('auc', 0):.4f}")

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, 'best_model.pt', val_metrics)
                print(f"  Saved best model (acc={self.best_val_acc:.4f})")

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Save final checkpoint
        self.save_checkpoint(epoch, 'final_model.pt', val_metrics)

        # Save training history to CSV
        self.save_history_csv()

        print(f"\nTraining complete! Best val accuracy: {self.best_val_acc:.4f}")
        print(f"Training metrics saved to: {self.save_dir / 'training_history.csv'}")

    def save_checkpoint(self, epoch: int, filename: str, metrics: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)

    def save_history_csv(self):
        """Save training history to CSV file"""
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'train_acc': self.history['train_acc'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc']
        })
        csv_path = self.save_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
