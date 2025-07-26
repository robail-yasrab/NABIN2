import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import json


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiOmicTrainer:
    """Trainer class for multi-omic transformer model."""
    
    def __init__(self, model: nn.Module, device: str = 'auto', 
                 use_focal_loss: bool = True, class_weights: Optional[torch.Tensor] = None):
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        
        # Loss functions
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.bottleneck_loss = nn.BCEWithLogitsLoss()  # For bottleneck identification
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_cls_loss': [],
            'train_btl_loss': [],
            'val_cls_loss': [],
            'val_btl_loss': []
        }
        
        print(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   bottleneck_weight: float = 0.3) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_btl_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Classification loss
            cls_loss = self.classification_loss(
                outputs['classification_logits'], 
                batch['outcome']
            )
            
            # Bottleneck identification loss (create pseudo labels based on importance)
            bottleneck_scores = outputs['bottleneck_scores']
            
            # Create bottleneck labels (top 10% of genes based on attention)
            mrna_attention = outputs['modality_attention']['mrna']
            top_k = max(1, mrna_attention.shape[1] // 10)  # Top 10%
            _, top_indices = torch.topk(mrna_attention, top_k, dim=1)
            
            bottleneck_labels = torch.zeros_like(bottleneck_scores)
            for i in range(bottleneck_labels.shape[0]):
                bottleneck_labels[i, top_indices[i]] = 1.0
            
            btl_loss = self.bottleneck_loss(bottleneck_scores, bottleneck_labels)
            
            # Combined loss
            total_loss_batch = cls_loss + bottleneck_weight * btl_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_btl_loss += btl_loss.item()
            
            predictions = torch.argmax(outputs['classification_logits'], dim=1)
            correct_predictions += (predictions == batch['outcome']).sum().item()
            total_samples += batch['outcome'].size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f"{total_loss_batch.item():.4f}",
                'Acc': f"{current_acc:.4f}"
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples,
            'cls_loss': total_cls_loss / len(train_loader),
            'btl_loss': total_btl_loss / len(train_loader)
        }
    
    def validate_epoch(self, val_loader: DataLoader, 
                      bottleneck_weight: float = 0.3) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_btl_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Classification loss
                cls_loss = self.classification_loss(
                    outputs['classification_logits'], 
                    batch['outcome']
                )
                
                # Bottleneck loss
                bottleneck_scores = outputs['bottleneck_scores']
                mrna_attention = outputs['modality_attention']['mrna']
                top_k = max(1, mrna_attention.shape[1] // 10)
                _, top_indices = torch.topk(mrna_attention, top_k, dim=1)
                
                bottleneck_labels = torch.zeros_like(bottleneck_scores)
                for i in range(bottleneck_labels.shape[0]):
                    bottleneck_labels[i, top_indices[i]] = 1.0
                
                btl_loss = self.bottleneck_loss(bottleneck_scores, bottleneck_labels)
                
                # Combined loss
                total_loss_batch = cls_loss + bottleneck_weight * btl_loss
                
                # Statistics
                total_loss += total_loss_batch.item()
                total_cls_loss += cls_loss.item()
                total_btl_loss += btl_loss.item()
                
                predictions = torch.argmax(outputs['classification_logits'], dim=1)
                probs = torch.softmax(outputs['classification_logits'], dim=1)
                
                correct_predictions += (predictions == batch['outcome']).sum().item()
                total_samples += batch['outcome'].size(0)
                
                # Store for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['outcome'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate additional metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct_predictions / total_samples,
            'cls_loss': total_cls_loss / len(val_loader),
            'btl_loss': total_btl_loss / len(val_loader)
        }
        
        # Add ROC AUC if binary classification
        if len(np.unique(all_labels)) == 2:
            try:
                auc_score = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
                metrics['auc'] = auc_score
            except:
                metrics['auc'] = 0.0
        
        return metrics, all_predictions, all_labels, all_probs
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 1e-4, weight_decay: float = 1e-5,
              bottleneck_weight: float = 0.3, save_dir: str = 'checkpoints',
              early_stopping_patience: int = 15) -> Dict[str, List]:
        """Complete training loop."""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, bottleneck_weight)
            
            # Validation
            val_metrics, val_preds, val_labels, val_probs = self.validate_epoch(
                val_loader, bottleneck_weight
            )
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_cls_loss'].append(train_metrics['cls_loss'])
            self.history['train_btl_loss'].append(train_metrics['btl_loss'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_cls_loss'].append(val_metrics['cls_loss'])
            self.history['val_btl_loss'].append(val_metrics['btl_loss'])
            
            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            if 'auc' in val_metrics:
                print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy']
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print("Saved best model!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model and history
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        all_bottleneck_scores = []
        all_sample_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                
                predictions = torch.argmax(outputs['classification_logits'], dim=1)
                probs = torch.softmax(outputs['classification_logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['outcome'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_bottleneck_scores.extend(outputs['bottleneck_scores'].cpu().numpy())
                all_sample_ids.extend(batch['sample_id'])
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'bottleneck_scores': all_bottleneck_scores,
            'sample_ids': all_sample_ids
        }
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions, 
            target_names=['Class_0', 'Class_1'], 
            output_dict=True
        )
        results['classification_report'] = report
        
        # ROC AUC for binary classification
        if len(np.unique(all_labels)) == 2:
            try:
                auc_score = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
                results['auc'] = auc_score
            except:
                results['auc'] = 0.0
        
        print(f"Test Accuracy: {accuracy:.4f}")
        if 'auc' in results:
            print(f"Test AUC: {results['auc']:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Classification loss
        axes[1, 0].plot(self.history['train_cls_loss'], label='Train Cls Loss')
        axes[1, 0].plot(self.history['val_cls_loss'], label='Val Cls Loss')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Bottleneck loss
        axes[1, 1].plot(self.history['train_btl_loss'], label='Train Btl Loss')
        axes[1, 1].plot(self.history['val_btl_loss'], label='Val Btl Loss')
        axes[1, 1].set_title('Bottleneck Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, test_results: Dict, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        
        cm = confusion_matrix(test_results['labels'], test_results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class_0', 'Class_1'], 
                   yticklabels=['Class_0', 'Class_1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_bottleneck_analysis(self, test_results: Dict, gene_names: List[str], 
                                save_path: str):
        """Save bottleneck gene analysis results."""
        
        bottleneck_scores = np.array(test_results['bottleneck_scores'])
        
        # Average scores across samples
        avg_scores = np.mean(bottleneck_scores, axis=0)
        
        # Create dataframe
        results_df = pd.DataFrame({
            'Gene': gene_names[:len(avg_scores)],
            'Bottleneck_Score': avg_scores
        })
        
        # Sort by score
        results_df = results_df.sort_values('Bottleneck_Score', ascending=False)
        
        # Save to CSV
        results_df.to_csv(save_path, index=False)
        print(f"Bottleneck analysis saved to {save_path}")
        
        return results_df


# Example usage and configuration class
class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        self.epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.bottleneck_weight = 0.3
        self.early_stopping_patience = 15
        self.use_focal_loss = True
        self.device = 'auto'
    
    def to_dict(self):
        return {
            'epochs': self.epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'bottleneck_weight': self.bottleneck_weight,
            'early_stopping_patience': self.early_stopping_patience,
            'use_focal_loss': self.use_focal_loss,
            'device': self.device
        }


# Helper function for easy training setup
def setup_training(model: nn.Module, config: TrainingConfig) -> MultiOmicTrainer:
    """Setup trainer with configuration."""
    
    trainer = MultiOmicTrainer(
        model=model,
        device=config.device,
        use_focal_loss=config.use_focal_loss
    )
    
    return trainer


if __name__ == "__main__":
    print("MultiOmic Trainer module loaded successfully!")
    print("Use TrainingConfig class to set up training parameters.")
    print("Example usage:")
    print("config = TrainingConfig()")
    print("trainer = setup_training(model, config)")
    print("history = trainer.train(train_loader, val_loader, **config.to_dict())") 