# Multi-Omic Metabolic Pathway Bottleneck Analysis

A transformer-based deep learning framework for integrating Copy Number Alterations (CNA), mutations, mRNA expression, and clinical timepoints to identify genes controlling bottlenecks in metabolic pathways.

## Features

- **Multi-modal data integration**: CNA, mutations, mRNA expression, clinical timepoints
- **Transformer-based architecture**: Advanced attention mechanisms for multi-omic fusion
- **Pathway-aware analysis**: Focus on metabolic pathways and bottleneck identification
- **Clinical temporal modeling**: Integration of time-series clinical data
- **Interpretable results**: Attention weights and pathway importance scores

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.multimodal_transformer import MultiOmicTransformer
from src.data.data_loader import MultiOmicDataLoader
from src.training.trainer import MultiOmicTrainer

# Load your data
data_loader = MultiOmicDataLoader()
train_loader, val_loader = data_loader.prepare_dataloaders()

# Initialize model
model = MultiOmicTransformer(
    cna_dim=1000,
    mutation_dim=1000, 
    mrna_dim=1000,
    clinical_dim=50,
    hidden_dim=256
)

# Train model
trainer = MultiOmicTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)
```

## Project Structure

```
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Neural network models
│   ├── training/       # Training utilities
│   ├── analysis/       # Pathway analysis tools
│   └── utils/          # Helper functions
├── examples/           # Example scripts and data
├── configs/           # Configuration files
└── notebooks/         # Jupyter notebooks for analysis
```

## Usage

See `examples/run_analysis.py` for a complete example with synthetic data. 