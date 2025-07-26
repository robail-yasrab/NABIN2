import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiModalAttention(nn.Module):
    """Multi-modal attention mechanism for cross-modal interactions."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super(MultiModalAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def attention(self, query, key, value, mask=None):
        """Scaled dot-product attention."""
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output), attn_weights


class OmicEncoder(nn.Module):
    """Encoder for individual omic data types."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(OmicEncoder, self).__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Project to hidden dimension
        h = self.input_projection(x)
        
        # Apply feature attention
        attention_weights = self.feature_attention(h)
        h_attended = h * attention_weights
        
        # Final projection
        output = self.output_projection(h_attended)
        
        return output, attention_weights


class TemporalEncoder(nn.Module):
    """Encoder for temporal clinical data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super(TemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Attention over time steps
        attention_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum over time
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Final projection
        output = self.output_projection(attended_output)  # [batch_size, hidden_dim]
        
        return output, attention_weights.squeeze(-1)


class PathwayAwareTransformer(nn.Module):
    """Transformer with pathway-aware attention."""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, 
                 pathway_info: Dict[str, list], dropout: float = 0.1):
        super(PathwayAwareTransformer, self).__init__()
        
        self.d_model = d_model
        self.pathway_info = pathway_info
        
        # Multi-modal attention layers
        self.attention_layers = nn.ModuleList([
            MultiModalAttention(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_layers)
        ])
        
        # Pathway attention
        self.pathway_attention = nn.ModuleDict({
            pathway: nn.Linear(d_model, 1) 
            for pathway in pathway_info.keys()
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def create_pathway_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """Create pathway-aware attention masks."""
        
        masks = {}
        for pathway_name, gene_indices in self.pathway_info.items():
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in gene_indices:
                for j in gene_indices:
                    if i < seq_len and j < seq_len:
                        mask[i, j] = 1
            masks[pathway_name] = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return masks
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Create pathway masks
        pathway_masks = self.create_pathway_mask(batch_size, seq_len, x.device)
        
        attention_weights_all = []
        
        # Apply transformer layers
        for i, (attn_layer, norm_layer, ff_layer) in enumerate(
            zip(self.attention_layers, self.layer_norms, self.feed_forwards)
        ):
            # Self-attention
            attn_output, attn_weights = attn_layer(x, x, x)
            x = norm_layer(x + self.dropout(attn_output))
            
            # Feed-forward
            ff_output = ff_layer(x)
            x = norm_layer(x + self.dropout(ff_output))
            
            if return_attention:
                attention_weights_all.append(attn_weights)
        
        # Pathway-specific attention
        pathway_outputs = {}
        for pathway_name, gene_indices in self.pathway_info.items():
            if len(gene_indices) == 0:
                continue
                
            # Extract pathway features
            valid_indices = [i for i in gene_indices if i < seq_len]
            if valid_indices:
                pathway_features = x[:, valid_indices, :]  # [batch_size, n_genes, d_model]
                
                # Apply pathway attention
                pathway_attn_scores = self.pathway_attention[pathway_name](pathway_features)
                pathway_attn_weights = F.softmax(pathway_attn_scores, dim=1)
                
                # Weighted sum
                pathway_output = torch.sum(pathway_features * pathway_attn_weights, dim=1)
                pathway_outputs[pathway_name] = pathway_output
        
        if return_attention:
            return x, pathway_outputs, attention_weights_all
        else:
            return x, pathway_outputs


class MultiOmicTransformer(nn.Module):
    """Complete multi-omic transformer model."""
    
    def __init__(self, cna_dim: int, mutation_dim: int, mrna_dim: int, 
                 clinical_dim: int, hidden_dim: int = 256, n_heads: int = 8, 
                 n_layers: int = 4, n_classes: int = 2, pathway_info: Optional[Dict] = None,
                 use_temporal: bool = False, dropout: float = 0.1):
        super(MultiOmicTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        self.pathway_info = pathway_info or {}
        
        # Omic encoders
        self.cna_encoder = OmicEncoder(cna_dim, hidden_dim, dropout)
        self.mutation_encoder = OmicEncoder(mutation_dim, hidden_dim, dropout)
        self.mrna_encoder = OmicEncoder(mrna_dim, hidden_dim, dropout)
        
        # Clinical encoder
        if use_temporal:
            self.clinical_encoder = TemporalEncoder(clinical_dim, hidden_dim, dropout=dropout)
        else:
            self.clinical_encoder = OmicEncoder(clinical_dim, hidden_dim, dropout)
        
        # Modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Pathway-aware transformer
        if pathway_info:
            self.pathway_transformer = PathwayAwareTransformer(
                hidden_dim, n_heads, n_layers, pathway_info, dropout
            )
        else:
            self.pathway_transformer = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Bottleneck identification head
        self.bottleneck_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, mrna_dim)  # Score for each gene
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch, return_attention=False):
        """Forward pass through the model."""
        
        # Extract data
        cna_data = batch['cna']
        mutation_data = batch['mutations']
        mrna_data = batch['mrna']
        clinical_data = batch['clinical']
        
        batch_size = cna_data.size(0)
        
        # Encode each modality
        cna_encoded, cna_attention = self.cna_encoder(cna_data)
        mutation_encoded, mutation_attention = self.mutation_encoder(mutation_data)
        mrna_encoded, mrna_attention = self.mrna_encoder(mrna_data)
        
        if self.use_temporal and clinical_data.dim() > 2:
            clinical_encoded, clinical_attention = self.clinical_encoder(clinical_data)
        else:
            clinical_encoded, clinical_attention = self.clinical_encoder(clinical_data)
        
        # Concatenate all modalities
        multimodal_features = torch.cat([
            cna_encoded, mutation_encoded, mrna_encoded, clinical_encoded
        ], dim=1)  # [batch_size, hidden_dim * 4]
        
        # Fuse modalities
        fused_features = self.modal_fusion(multimodal_features)  # [batch_size, hidden_dim]
        
        # Pathway-aware processing
        pathway_outputs = {}
        transformer_attention = None
        
        if self.pathway_transformer is not None:
            # Create gene-level features for pathway analysis
            gene_features = torch.cat([
                cna_encoded.unsqueeze(1),
                mutation_encoded.unsqueeze(1), 
                mrna_encoded.unsqueeze(1)
            ], dim=1)  # [batch_size, 3, hidden_dim]
            
            if return_attention:
                _, pathway_outputs, transformer_attention = self.pathway_transformer(
                    gene_features, return_attention=True
                )
            else:
                _, pathway_outputs = self.pathway_transformer(gene_features)
        
        # Classification
        classification_logits = self.classifier(fused_features)
        
        # Bottleneck identification
        bottleneck_scores = self.bottleneck_head(fused_features)
        
        outputs = {
            'classification_logits': classification_logits,
            'bottleneck_scores': bottleneck_scores,
            'fused_features': fused_features,
            'pathway_outputs': pathway_outputs,
            'modality_attention': {
                'cna': cna_attention,
                'mutations': mutation_attention,
                'mrna': mrna_attention,
                'clinical': clinical_attention
            }
        }
        
        if return_attention and transformer_attention is not None:
            outputs['transformer_attention'] = transformer_attention
        
        return outputs
    
    def get_gene_importance_scores(self, batch):
        """Get gene importance scores from the model."""
        
        with torch.no_grad():
            outputs = self.forward(batch, return_attention=True)
            
            # Combine bottleneck scores and attention weights
            bottleneck_scores = outputs['bottleneck_scores']
            mrna_attention = outputs['modality_attention']['mrna']
            
            # Weighted combination
            importance_scores = bottleneck_scores * mrna_attention
            
            return importance_scores.cpu().numpy()
    
    def get_pathway_importance_scores(self, batch):
        """Get pathway-level importance scores."""
        
        with torch.no_grad():
            outputs = self.forward(batch)
            pathway_outputs = outputs['pathway_outputs']
            
            # Calculate pathway importance based on activation magnitude
            pathway_importance = {}
            for pathway_name, pathway_features in pathway_outputs.items():
                importance = torch.norm(pathway_features, dim=1).mean().item()
                pathway_importance[pathway_name] = importance
            
            return pathway_importance


# Helper function for model initialization
def create_multimodal_transformer(dataset_info: Dict, pathway_info: Dict, 
                                config: Dict) -> MultiOmicTransformer:
    """Create and initialize the multi-omic transformer model."""
    
    model = MultiOmicTransformer(
        cna_dim=dataset_info['cna_dim'],
        mutation_dim=dataset_info['mutation_dim'],
        mrna_dim=dataset_info['mrna_dim'],
        clinical_dim=dataset_info['clinical_dim'],
        hidden_dim=config.get('hidden_dim', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 4),
        n_classes=dataset_info['n_classes'],
        pathway_info=pathway_info,
        use_temporal=config.get('use_temporal', False),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Example configuration
    dataset_info = {
        'cna_dim': 1000,
        'mutation_dim': 1000,
        'mrna_dim': 1000,
        'clinical_dim': 8,
        'n_classes': 2
    }
    
    pathway_info = {
        'Glycolysis': list(range(0, 100)),
        'TCA_Cycle': list(range(100, 200)),
        'Oxidative_Phosphorylation': list(range(200, 300))
    }
    
    config = {
        'hidden_dim': 256,
        'n_heads': 8,
        'n_layers': 4,
        'use_temporal': False,
        'dropout': 0.1
    }
    
    # Create model
    model = create_multimodal_transformer(dataset_info, pathway_info, config)
    
    # Example batch
    batch = {
        'cna': torch.randn(16, 1000),
        'mutations': torch.randn(16, 1000),
        'mrna': torch.randn(16, 1000),
        'clinical': torch.randn(16, 8)
    }
    
    # Forward pass
    outputs = model(batch)
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}") 