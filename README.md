# Multi-Omics Metabolic Pathway Bottleneck Gene Analysis

🧬 **Advanced AI/ML solution for integrating multi-omics data to identify bottleneck genes in metabolic pathways**

## Overview

This project implements a state-of-the-art **Multimodal Variational Autoencoder (MVAE)** combined with **Graph Neural Networks** to integrate multiple types of genomic data and identify genes that control bottlenecks in metabolic pathways. The approach can handle real clinical timepoints and predict outcomes while discovering pathway control points.

### Key Features

- 🔬 **Multi-Omics Integration**: CNA, mutations, mRNA expression, and clinical timepoints
- 🧠 **Advanced AI Architecture**: Multimodal VAE + Graph Attention Networks
- 🎯 **Bottleneck Identification**: Attention-based mechanism to find pathway control genes
- 📊 **Clinical Prediction**: Survival analysis and treatment response prediction
- 📈 **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, clinical metrics
- 🎨 **Rich Visualizations**: Interactive plots and pathway network analysis

## Methodology

### 1. Data Integration Strategy

Our approach integrates four types of data:

- **Copy Number Alterations (CNA)**: Discrete values (-2 to +2)
- **Mutation Data**: Binary indicators of genetic mutations
- **mRNA Expression**: Continuous gene expression levels
- **Clinical Timepoints**: Survival, progression-free survival, response scores

### 2. Model Architecture

```
Input Modalities → Individual Encoders → Product of Experts → Shared Latent Space
                                                                      ↓
                                                          Graph Neural Network
                                                                      ↓
                                                          Attention Mechanism
                                                                      ↓
                                              Bottleneck Identification + Clinical Prediction
```

#### Key Components:

1. **Modality-Specific Encoders**: Separate VAE encoders for each data type
2. **Product of Experts**: Probabilistic fusion of modality representations
3. **Graph Attention Networks**: Model metabolic pathway structure
4. **Attention-Based Bottleneck Identifier**: Multi-head attention to find important genes
5. **Clinical Outcome Predictors**: Multi-task learning for survival and response

### 3. Loss Function

The model optimizes a weighted combination of losses:

```
L_total = α·L_reconstruction + β·L_KL + γ·L_clinical + δ·L_bottleneck + ε·L_pathway
```

Where:
- **L_reconstruction**: MSE for continuous data, BCE for binary data
- **L_KL**: KL divergence for variational inference
- **L_clinical**: MSE for survival/response prediction
- **L_bottleneck**: BCE for bottleneck gene identification
- **L_pathway**: Contrastive loss for pathway structure

## Installation

### Requirements

```bash
# Create conda environment
conda create -n multiomics python=3.9
conda activate multiomics

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch ≥ 2.0.0
- PyTorch Geometric ≥ 2.3.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly
- NetworkX, SciPy

## Usage

### Quick Start

```bash
# Run complete pipeline with default settings
python main.py --mode full --num_epochs 50

# Custom configuration
python main.py --mode full \
    --n_samples 1000 \
    --n_genes 500 \
    --n_pathways 20 \
    --latent_dim 128 \
    --batch_size 32 \
    --num_epochs 100 \
    --output_dir ./results/
```

### Step-by-Step Execution

1. **Data Generation Only**:
```bash
python main.py --mode data_only --n_samples 1000 --n_genes 500
```

2. **Training Only** (requires existing dataset):
```bash
python main.py --mode train_only --load_data ./results/multiomics_dataset.pt
```

3. **Evaluation Only** (requires trained model):
```bash
python main.py --mode eval_only --load_model ./results/multiomics_model.pth
```

### Advanced Usage

#### Custom Loss Weights
```bash
python main.py --mode full \
    --recon_weight 1.0 \
    --kl_weight 0.1 \
    --clinical_weight 0.5 \
    --bottleneck_weight 0.3 \
    --pathway_weight 0.2
```

#### Different Data Sizes
```bash
# Small dataset for testing
python main.py --n_samples 500 --n_genes 200 --n_pathways 10

# Large dataset for production
python main.py --n_samples 5000 --n_genes 1000 --n_pathways 50
```

## File Structure

```
├── main.py                 # Main execution script
├── data_generator.py       # Synthetic multi-omics data generation
├── multimodal_models.py    # MVAE + GNN architecture
├── training.py            # Training pipeline and loss functions
├── evaluation.py          # Evaluation metrics and visualizations
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── results/               # Output directory
    ├── multiomics_dataset.pt      # Generated dataset
    ├── multiomics_model.pth       # Trained model
    ├── training_history.png       # Training curves
    ├── bottleneck_analysis.png    # Bottleneck gene analysis
    ├── pathway_analysis.png       # Pathway network visualization
    ├── clinical_predictions.png   # Clinical outcome predictions
    └── evaluation_metrics.json    # Comprehensive metrics
```

## Output Interpretation

### 1. Bottleneck Gene Identification

The model outputs bottleneck scores for each gene (0-1 scale):
- **High scores (>0.7)**: Likely bottleneck genes
- **Medium scores (0.3-0.7)**: Potential pathway regulators
- **Low scores (<0.3)**: Regular pathway genes

### 2. Key Metrics

- **ROC-AUC**: Overall bottleneck identification performance
- **Precision@K**: Accuracy of top-K predictions
- **Clinical R²**: Quality of survival/response predictions
- **Reconstruction MSE**: Data integration quality

### 3. Visualizations

1. **Bottleneck Analysis**: ROC/PR curves, score distributions
2. **Pathway Networks**: Graph visualization with bottleneck highlighting
3. **Clinical Predictions**: True vs. predicted outcomes
4. **Training History**: Loss curves and convergence analysis

## Model Performance

### Expected Results (Synthetic Data)

- **Bottleneck Identification ROC-AUC**: 0.85-0.95
- **Precision@20**: 0.6-0.8
- **Clinical Prediction R²**: 0.4-0.7
- **Reconstruction Quality**: MSE < 0.1

### Hyperparameter Sensitivity

| Parameter | Recommended Range | Impact |
|-----------|------------------|--------|
| `latent_dim` | 64-256 | Model capacity |
| `kl_weight` | 0.01-0.5 | Regularization strength |
| `bottleneck_weight` | 0.1-0.5 | Bottleneck focus |
| `learning_rate` | 1e-4-1e-2 | Convergence speed |

## Real Data Adaptation

To use with real multi-omics data:

1. **Data Preprocessing**:
   - Normalize expression data (log2 transform)
   - Encode CNA as integers (-2 to +2)
   - Binarize mutation data
   - Handle missing values

2. **Pathway Information**:
   - Use databases like KEGG, Reactome, or BioCyc
   - Create NetworkX graph from pathway annotations

3. **Ground Truth**:
   - Literature-based bottleneck gene lists
   - Experimental validation data
   - Known pathway control points

## Extending the Model

### Adding New Modalities

```python
# In multimodal_models.py
self.new_modality_encoder = Encoder(
    input_dim=new_modality_dim,
    hidden_dims=[512, 256],
    latent_dim=config['latent_dim']
)
```

### Custom Loss Functions

```python
# In training.py
def custom_pathway_loss(self, gene_embeddings, custom_graph):
    # Implement domain-specific pathway constraints
    pass
```

### Different Clinical Outcomes

```python
# Add new predictors in multimodal_models.py
self.new_outcome_predictor = nn.Sequential(
    nn.Linear(config['latent_dim'], config['clinical_hidden_dim']),
    nn.ReLU(),
    nn.Linear(config['clinical_hidden_dim'], output_dim)
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size` or `latent_dim`
   - Use gradient checkpointing

2. **Poor Convergence**:
   - Adjust learning rate or loss weights
   - Increase model capacity
   - Check data normalization

3. **Overfitting**:
   - Increase dropout rates
   - Add more regularization
   - Reduce model complexity

### Performance Optimization

- Use mixed precision training: `--amp`
- Enable data parallelism for large datasets
- Profile bottlenecks with PyTorch profiler

## Scientific Background

### Bottleneck Genes in Metabolism

Bottleneck genes are crucial regulatory points in metabolic pathways that:
- Control pathway flux
- Determine cellular responses to perturbations
- Represent potential therapeutic targets
- Show evolutionary conservation

### Multi-Omics Integration Challenges

1. **Different data types**: Continuous vs. discrete vs. binary
2. **Scale differences**: Expression vs. copy number vs. mutations
3. **Missing data**: Incomplete measurements across modalities
4. **Temporal dynamics**: Clinical timepoints vs. static genomics

### VAE Advantages for Genomics

- **Uncertainty quantification**: Probabilistic representations
- **Dimensionality reduction**: Handle high-dimensional genomic data
- **Missing data handling**: Reconstruction capabilities
- **Interpretability**: Latent space analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multiomics_bottleneck_analysis,
  title={Multi-Omics Metabolic Pathway Bottleneck Gene Analysis},
  author={AI Assistant},
  year={2025},
  url={https://github.com/your-repo/multiomics-analysis}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by multimodal VAE research from [mhw32/multimodal-vae-public](https://github.com/mhw32/multimodal-vae-public)
- Graph neural network implementations from PyTorch Geometric
- Metabolic pathway databases: KEGG, Reactome, BioCyc

---

**Contact**: For questions or collaborations, please open an issue or contact the development team.

🚀 **Ready to discover metabolic bottlenecks in your data? Get started with `python main.py --mode full`!** 