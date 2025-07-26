import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch_geometric.utils as pyg_utils

class Encoder(nn.Module):
    """Generic encoder for different data modalities."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.2):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        Returns: mu, logvar, z (reparameterized sample)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z

class Decoder(nn.Module):
    """Generic decoder for different data modalities."""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, 
                 output_activation: str = 'linear', dropout: float = 0.2):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        h = self.decoder(z)
        output = self.output_layer(h)
        
        if self.output_activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.output_activation == 'tanh':
            output = torch.tanh(output)
        elif self.output_activation == 'softmax':
            output = F.softmax(output, dim=-1)
        
        return output

class PathwayGraphEncoder(nn.Module):
    """Graph neural network encoder for pathway structure."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 3):
        super(PathwayGraphEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=0.2))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=0.2))
        
        self.gat_layers.append(GATConv(hidden_dim * 4, latent_dim, heads=1, concat=False, dropout=0.2))
        
        # Global pooling for graph-level representation
        self.global_pool = global_mean_pool
        
        # Final projection layers
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph encoder.
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
        """
        h = x
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
        
        # Global pooling if batch is provided (for batched graphs)
        if batch is not None:
            h = self.global_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)  # Simple mean pooling for single graph
        
        # Latent distribution parameters
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z

class AttentionBottleneckIdentifier(nn.Module):
    """Attention mechanism to identify bottleneck genes."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(AttentionBottleneckIdentifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-head attention for gene importance
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # Bottleneck scoring network
        self.bottleneck_scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Pathway interaction network
        self.pathway_interaction = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, gene_embeddings: torch.Tensor, pathway_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Identify bottleneck genes using attention mechanism.
        Args:
            gene_embeddings: [batch_size, num_genes, embedding_dim]
            pathway_mask: [batch_size, num_genes, num_pathways] optional pathway membership
        """
        batch_size, num_genes, embed_dim = gene_embeddings.shape
        
        # Self-attention to capture gene interactions
        attn_output, attn_weights = self.multihead_attn(gene_embeddings, gene_embeddings, gene_embeddings)
        
        # Score each gene for bottleneck importance
        bottleneck_scores = self.bottleneck_scorer(attn_output).squeeze(-1)  # [batch_size, num_genes]
        
        # Compute pairwise gene interactions
        gene_pairs = []
        interaction_scores = []
        
        for i in range(num_genes):
            for j in range(i + 1, num_genes):
                gene_pair = torch.cat([gene_embeddings[:, i, :], gene_embeddings[:, j, :]], dim=-1)
                interaction_score = self.pathway_interaction(gene_pair)
                gene_pairs.append((i, j))
                interaction_scores.append(interaction_score)
        
        interaction_tensor = torch.stack(interaction_scores, dim=1).squeeze(-1)  # [batch_size, num_pairs]
        
        return {
            'bottleneck_scores': bottleneck_scores,
            'attention_weights': attn_weights,
            'interaction_scores': interaction_tensor,
            'gene_pairs': gene_pairs,
            'attended_embeddings': attn_output
        }

class MultimodalVAE(nn.Module):
    """Multimodal Variational Autoencoder for integrating multi-omics data."""
    
    def __init__(self, config: Dict):
        super(MultimodalVAE, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.n_genes = config['n_genes']
        
        # Modality-specific encoders
        self.mrna_encoder = Encoder(
            input_dim=config['n_genes'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim']
        )
        
        self.cna_encoder = Encoder(
            input_dim=config['n_genes'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim']
        )
        
        self.mutation_encoder = Encoder(
            input_dim=config['n_genes'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim']
        )
        
        # Graph encoder for pathway structure
        self.pathway_encoder = PathwayGraphEncoder(
            input_dim=config['latent_dim'],
            hidden_dim=config['graph_hidden_dim'],
            latent_dim=config['latent_dim']
        )
        
        # Shared latent space encoder
        self.shared_encoder = Encoder(
            input_dim=config['latent_dim'] * 3,  # Concatenated modalities
            hidden_dims=config['shared_hidden_dims'],
            latent_dim=config['latent_dim']
        )
        
        # Modality-specific decoders
        self.mrna_decoder = Decoder(
            latent_dim=config['latent_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            output_dim=config['n_genes'],
            output_activation='linear'
        )
        
        self.cna_decoder = Decoder(
            latent_dim=config['latent_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            output_dim=config['n_genes'],
            output_activation='tanh'  # CNA values typically in [-2, 2]
        )
        
        self.mutation_decoder = Decoder(
            latent_dim=config['latent_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            output_dim=config['n_genes'],
            output_activation='sigmoid'  # Binary mutations
        )
        
        # Clinical outcome predictors
        self.survival_predictor = nn.Sequential(
            nn.Linear(config['latent_dim'], config['clinical_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['clinical_hidden_dim'], 1)
        )
        
        self.response_predictor = nn.Sequential(
            nn.Linear(config['latent_dim'], config['clinical_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['clinical_hidden_dim'], 1),
            nn.Sigmoid()
        )
        
        # Bottleneck gene identifier
        self.bottleneck_identifier = AttentionBottleneckIdentifier(
            input_dim=config['latent_dim']
        )
        
        # Product of experts for multimodal fusion
        self.fusion_weights = nn.Parameter(torch.ones(4))  # 3 modalities + pathway
    
    def encode_modalities(self, data: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode all modalities separately."""
        encodings = {}
        
        # Encode individual modalities
        encodings['mrna'] = self.mrna_encoder(data['mrna_expression'])
        encodings['cna'] = self.cna_encoder(data['cna'])
        encodings['mutations'] = self.mutation_encoder(data['mutations'])
        
        return encodings
    
    def product_of_experts(self, mus: List[torch.Tensor], logvars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine multiple Gaussian distributions using product of experts."""
        # Convert logvars to precisions
        precisions = [torch.exp(-logvar) for logvar in logvars]
        
        # Weighted precisions
        weighted_precisions = []
        for i, precision in enumerate(precisions):
            weight = F.softmax(self.fusion_weights, dim=0)[i]
            weighted_precisions.append(weight * precision)
        
        # Combined precision and mean
        combined_precision = sum(weighted_precisions)
        combined_var = 1.0 / combined_precision
        
        weighted_mus = []
        for i, (mu, precision) in enumerate(zip(mus, precisions)):
            weight = F.softmax(self.fusion_weights, dim=0)[i]
            weighted_mus.append(weight * precision * mu)
        
        combined_mu = combined_var * sum(weighted_mus)
        combined_logvar = torch.log(combined_var)
        
        return combined_mu, combined_logvar
    
    def forward(self, data: Dict[str, torch.Tensor], pathway_graph: Optional[Data] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the multimodal VAE."""
        batch_size = data['mrna_expression'].shape[0]
        
        # Encode modalities
        modality_encodings = self.encode_modalities(data)
        
        # Extract latent representations
        mus = [encoding[0] for encoding in modality_encodings.values()]
        logvars = [encoding[1] for encoding in modality_encodings.values()]
        zs = [encoding[2] for encoding in modality_encodings.values()]
        
        # Encode pathway structure if available
        if pathway_graph is not None:
            # Use the mean of modality representations as node features
            node_features = torch.stack(zs, dim=-1).mean(dim=-1)  # Average across modalities
            pathway_mu, pathway_logvar, pathway_z = self.pathway_encoder(
                node_features, pathway_graph.edge_index
            )
            mus.append(pathway_mu)
            logvars.append(pathway_logvar)
        
        # Fuse modalities using product of experts
        fused_mu, fused_logvar = self.product_of_experts(mus, logvars)
        
        # Sample from fused distribution
        std = torch.exp(0.5 * fused_logvar)
        eps = torch.randn_like(std)
        fused_z = fused_mu + eps * std
        
        # Decode to reconstruct modalities
        reconstructions = {
            'mrna_recon': self.mrna_decoder(fused_z),
            'cna_recon': self.cna_decoder(fused_z) * 2,  # Scale to [-2, 2] range
            'mutations_recon': self.mutation_decoder(fused_z)
        }
        
        # Predict clinical outcomes
        clinical_predictions = {
            'survival_pred': self.survival_predictor(fused_z).squeeze(-1),
            'response_pred': self.response_predictor(fused_z).squeeze(-1) * 100  # Scale to [0, 100]
        }
        
        # Create gene embeddings for bottleneck identification
        gene_embeddings = fused_z.unsqueeze(1).expand(-1, self.n_genes, -1)  # [batch_size, n_genes, latent_dim]
        
        # Identify bottleneck genes
        bottleneck_analysis = self.bottleneck_identifier(gene_embeddings)
        
        return {
            'reconstructions': reconstructions,
            'clinical_predictions': clinical_predictions,
            'bottleneck_analysis': bottleneck_analysis,
            'latent_representations': {
                'modality_mus': mus[:-1] if pathway_graph is not None else mus,  # Exclude pathway if present
                'modality_logvars': logvars[:-1] if pathway_graph is not None else logvars,
                'modality_zs': zs,
                'fused_mu': fused_mu,
                'fused_logvar': fused_logvar,
                'fused_z': fused_z
            }
        }

def create_model_config(n_genes: int = 500) -> Dict:
    """Create default model configuration."""
    return {
        'n_genes': n_genes,
        'latent_dim': 128,
        'encoder_hidden_dims': [512, 256],
        'decoder_hidden_dims': [256, 512],
        'shared_hidden_dims': [256, 128],
        'graph_hidden_dim': 64,
        'clinical_hidden_dim': 64
    }

if __name__ == "__main__":
    # Test the model
    config = create_model_config(n_genes=500)
    model = MultimodalVAE(config)
    
    # Create dummy data
    batch_size = 32
    n_genes = 500
    
    dummy_data = {
        'mrna_expression': torch.randn(batch_size, n_genes),
        'cna': torch.randint(-2, 3, (batch_size, n_genes)).float(),
        'mutations': torch.bernoulli(torch.full((batch_size, n_genes), 0.05))
    }
    
    # Forward pass
    output = model(dummy_data)
    
    print("Model test successful!")
    print(f"Fused latent dimension: {output['latent_representations']['fused_z'].shape}")
    print(f"Bottleneck scores shape: {output['bottleneck_analysis']['bottleneck_scores'].shape}")
    print(f"mRNA reconstruction shape: {output['reconstructions']['mrna_recon'].shape}") 