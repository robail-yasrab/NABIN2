import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import pandas as pd

from multimodal_models import MultimodalVAE, create_model_config
from data_generator import MultiOmicsDataGenerator

class MultiOmicsLoss(nn.Module):
    """Comprehensive loss function for multimodal multi-omics analysis."""
    
    def __init__(self, config: Dict):
        super(MultiOmicsLoss, self).__init__()
        self.config = config
        
        # Loss weights
        self.recon_weight = config.get('recon_weight', 1.0)
        self.kl_weight = config.get('kl_weight', 0.1)
        self.clinical_weight = config.get('clinical_weight', 0.5)
        self.bottleneck_weight = config.get('bottleneck_weight', 0.3)
        self.pathway_weight = config.get('pathway_weight', 0.2)
        
        # Reconstruction loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
        
    def reconstruction_loss(self, reconstructions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute reconstruction losses for each modality."""
        losses = {}
        
        # mRNA expression reconstruction (continuous data)
        losses['mrna_recon'] = self.mse_loss(
            reconstructions['mrna_recon'], 
            targets['mrna_expression']
        )
        
        # CNA reconstruction (discrete but treated as continuous)
        losses['cna_recon'] = self.huber_loss(
            reconstructions['cna_recon'], 
            targets['cna']
        )
        
        # Mutation reconstruction (binary data)
        losses['mutations_recon'] = self.bce_loss(
            reconstructions['mutations_recon'], 
            targets['mutations']
        )
        
        return losses
    
    def kl_divergence_loss(self, mus: List[torch.Tensor], 
                          logvars: List[torch.Tensor]) -> torch.Tensor:
        """Compute KL divergence loss for variational inference."""
        kl_loss = 0.0
        
        for mu, logvar in zip(mus, logvars):
            # KL(q(z|x) || p(z)) where p(z) is standard normal
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss += kl.mean()
        
        return kl_loss / len(mus)  # Average across modalities
    
    def clinical_prediction_loss(self, clinical_predictions: Dict[str, torch.Tensor],
                               clinical_targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute clinical outcome prediction losses."""
        losses = {}
        
        # Survival prediction (continuous)
        if 'survival_pred' in clinical_predictions and 'overall_survival' in clinical_targets:
            losses['survival'] = self.mse_loss(
                clinical_predictions['survival_pred'],
                clinical_targets['overall_survival']
            )
        
        # Response prediction (continuous, 0-100 scale)
        if 'response_pred' in clinical_predictions and 'response_score' in clinical_targets:
            losses['response'] = self.mse_loss(
                clinical_predictions['response_pred'],
                clinical_targets['response_score']
            )
        
        return losses
    
    def bottleneck_identification_loss(self, bottleneck_analysis: Dict[str, torch.Tensor],
                                     true_bottleneck_genes: List[int],
                                     n_genes: int) -> torch.Tensor:
        """Compute loss for bottleneck gene identification."""
        # Create ground truth bottleneck mask
        batch_size = bottleneck_analysis['bottleneck_scores'].shape[0]
        bottleneck_mask = torch.zeros(batch_size, n_genes)
        
        for gene_idx in true_bottleneck_genes:
            bottleneck_mask[:, gene_idx] = 1.0
        
        bottleneck_mask = bottleneck_mask.to(bottleneck_analysis['bottleneck_scores'].device)
        
        # Binary cross-entropy for bottleneck identification
        bottleneck_loss = self.bce_loss(
            bottleneck_analysis['bottleneck_scores'],
            bottleneck_mask
        )
        
        return bottleneck_loss
    
    def pathway_structure_loss(self, gene_embeddings: torch.Tensor,
                             pathway_graph: Optional[Data] = None) -> torch.Tensor:
        """Compute loss to encourage pathway structure in embeddings."""
        if pathway_graph is None:
            return torch.tensor(0.0, device=gene_embeddings.device)
        
        # Contrastive loss for genes in the same pathway
        edge_index = pathway_graph.edge_index
        
        # Compute pairwise distances for connected genes
        connected_distances = []
        for i in range(edge_index.shape[1]):
            gene1_idx, gene2_idx = edge_index[0, i], edge_index[1, i]
            dist = F.pairwise_distance(
                gene_embeddings[:, gene1_idx, :],
                gene_embeddings[:, gene2_idx, :]
            )
            connected_distances.append(dist)
        
        if connected_distances:
            connected_distances = torch.stack(connected_distances, dim=1)
            pathway_loss = connected_distances.mean()
        else:
            pathway_loss = torch.tensor(0.0, device=gene_embeddings.device)
        
        return pathway_loss
    
    def forward(self, model_output: Dict, targets: Dict, 
                true_bottleneck_genes: List[int], 
                pathway_graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """Compute total loss and individual components."""
        losses = {}
        
        # Reconstruction losses
        recon_losses = self.reconstruction_loss(
            model_output['reconstructions'], 
            targets
        )
        losses.update(recon_losses)
        total_recon_loss = sum(recon_losses.values())
        
        # KL divergence loss
        kl_loss = self.kl_divergence_loss(
            model_output['latent_representations']['modality_mus'],
            model_output['latent_representations']['modality_logvars']
        )
        losses['kl_divergence'] = kl_loss
        
        # Clinical prediction losses
        clinical_losses = self.clinical_prediction_loss(
            model_output['clinical_predictions'],
            targets
        )
        losses.update(clinical_losses)
        total_clinical_loss = sum(clinical_losses.values()) if clinical_losses else 0.0
        
        # Bottleneck identification loss
        bottleneck_loss = self.bottleneck_identification_loss(
            model_output['bottleneck_analysis'],
            true_bottleneck_genes,
            targets['mrna_expression'].shape[1]
        )
        losses['bottleneck_identification'] = bottleneck_loss
        
        # Pathway structure loss
        gene_embeddings = model_output['latent_representations']['fused_z'].unsqueeze(1).expand(
            -1, targets['mrna_expression'].shape[1], -1
        )
        pathway_loss = self.pathway_structure_loss(gene_embeddings, pathway_graph)
        losses['pathway_structure'] = pathway_loss
        
        # Total weighted loss
        total_loss = (
            self.recon_weight * total_recon_loss +
            self.kl_weight * kl_loss +
            self.clinical_weight * total_clinical_loss +
            self.bottleneck_weight * bottleneck_loss +
            self.pathway_weight * pathway_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses

class MultiOmicsTrainer:
    """Trainer class for the multimodal multi-omics model."""
    
    def __init__(self, model: MultimodalVAE, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = MultiOmicsLoss(config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, verbose=True
        )
        
        # Training history
        self.train_history = []
        self.val_history = []
    
    def prepare_data(self, dataset: Dict) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation."""
        # Convert data to tensors
        mrna_data = dataset['mrna_expression']
        cna_data = dataset['cna']
        mutations_data = dataset['mutations']
        survival_data = dataset['overall_survival']
        response_data = dataset['response_score']
        
        # Create tensor dataset
        tensor_dataset = TensorDataset(
            mrna_data, cna_data, mutations_data, 
            survival_data, response_data
        )
        
        # Split into train/validation
        train_size = int(0.8 * len(tensor_dataset))
        val_size = len(tensor_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            tensor_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def prepare_batch_data(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        """Convert batch tuple to dictionary format."""
        mrna, cna, mutations, survival, response = batch
        
        return {
            'mrna_expression': mrna.to(self.device),
            'cna': cna.to(self.device),
            'mutations': mutations.to(self.device),
            'overall_survival': survival.to(self.device),
            'response_score': response.to(self.device)
        }
    
    def train_epoch(self, train_loader: DataLoader, 
                   bottleneck_genes: List[int],
                   pathway_graph: Optional[Data] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Prepare batch data
            batch_data = self.prepare_batch_data(batch)
            
            # Forward pass
            model_output = self.model(batch_data, pathway_graph)
            
            # Compute losses
            losses = self.loss_fn(
                model_output, batch_data, 
                bottleneck_genes, pathway_graph
            )
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] = np.mean(epoch_losses[loss_name])
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader,
                      bottleneck_genes: List[int],
                      pathway_graph: Optional[Data] = None) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Prepare batch data
                batch_data = self.prepare_batch_data(batch)
                
                # Forward pass
                model_output = self.model(batch_data, pathway_graph)
                
                # Compute losses
                losses = self.loss_fn(
                    model_output, batch_data,
                    bottleneck_genes, pathway_graph
                )
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in epoch_losses:
                        epoch_losses[loss_name] = []
                    epoch_losses[loss_name].append(loss_value.item())
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] = np.mean(epoch_losses[loss_name])
        
        return epoch_losses
    
    def train(self, dataset: Dict, num_epochs: int = 100) -> Dict:
        """Full training loop."""
        print(f"Starting training on {self.device}")
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data(dataset)
        
        # Get bottleneck genes and pathway graph
        bottleneck_genes = dataset['bottleneck_genes']
        pathway_graph = self.convert_networkx_to_pyg(dataset['pathway_graph'])
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader, bottleneck_genes, pathway_graph)
            
            # Validate
            val_losses = self.validate_epoch(val_loader, bottleneck_genes, pathway_graph)
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Save history
            self.train_history.append(train_losses)
            self.val_history.append(val_losses)
            
            # Print losses
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            # Early stopping and model saving
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print("✓ New best model saved")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.get('patience', 20):
                print("Early stopping triggered")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': best_val_loss
        }
    
    def convert_networkx_to_pyg(self, nx_graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        # Get edge list
        edge_list = list(nx_graph.edges())
        if not edge_list:
            # Create empty graph with self-loops
            num_nodes = len(nx_graph.nodes())
            edge_index = torch.arange(num_nodes).repeat(2, 1)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Node features (will be filled during forward pass)
        num_nodes = len(nx_graph.nodes())
        x = torch.eye(num_nodes)  # Identity matrix as placeholder
        
        return Data(x=x, edge_index=edge_index).to(self.device)
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"Model loaded from {path}")

def plot_training_history(train_history: List[Dict], val_history: List[Dict], save_path: str = None):
    """Plot training and validation losses over epochs."""
    # Extract loss names
    loss_names = list(train_history[0].keys())
    
    # Create subplots
    n_losses = len(loss_names)
    n_cols = 3
    n_rows = (n_losses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, loss_name in enumerate(loss_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        train_values = [epoch_losses[loss_name] for epoch_losses in train_history]
        val_values = [epoch_losses[loss_name] for epoch_losses in val_history]
        
        ax.plot(train_values, label='Train', alpha=0.7)
        ax.plot(val_values, label='Validation', alpha=0.7)
        ax.set_title(f'{loss_name.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_losses, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic multi-omics dataset...")
    generator = MultiOmicsDataGenerator(n_samples=1000, n_genes=500, n_pathways=20)
    dataset = generator.generate_complete_dataset()
    
    # Create model configuration
    model_config = create_model_config(n_genes=500)
    model_config.update({
        'learning_rate': 1e-3,
        'batch_size': 32,
        'weight_decay': 1e-4,
        'patience': 20,
        'recon_weight': 1.0,
        'kl_weight': 0.1,
        'clinical_weight': 0.5,
        'bottleneck_weight': 0.3,
        'pathway_weight': 0.2
    })
    
    # Initialize model and trainer
    model = MultimodalVAE(model_config)
    trainer = MultiOmicsTrainer(model, model_config)
    
    # Train the model
    print("Starting training...")
    training_results = trainer.train(dataset, num_epochs=50)
    
    # Save model
    trainer.save_model('multiomics_model.pth')
    
    # Plot training history
    plot_training_history(
        training_results['train_history'],
        training_results['val_history'],
        'training_history.png'
    )
    
    print("Training completed successfully!") 