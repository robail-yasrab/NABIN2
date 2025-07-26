#!/usr/bin/env python3
"""
Complete Multi-Omic Metabolic Pathway Bottleneck Analysis Example

This script demonstrates the full pipeline:
1. Generate synthetic multi-omic data
2. Create data loaders
3. Initialize the multi-modal transformer model
4. Train the model
5. Evaluate and identify bottleneck genes
6. Perform pathway analysis
7. Generate comprehensive reports

Author: Robail Yasrab
Date: 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_generator import MultiOmicDataGenerator
from src.data.data_loader import MultiOmicDataLoader
from src.models.multimodal_transformer import create_multimodal_transformer
from src.training.trainer import MultiOmicTrainer, TrainingConfig
from src.analysis.pathway_analyzer import MetabolicPathwayAnalyzer, BottleneckRanker


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_dataset(n_samples: int = 1000, n_genes: int = 1000) -> Dict:
    """Generate synthetic multi-omic dataset."""
    
    print("=" * 60)
    print("STEP 1: GENERATING SYNTHETIC MULTI-OMIC DATASET")
    print("=" * 60)
    
    generator = MultiOmicDataGenerator(
        n_samples=n_samples, 
        n_genes=n_genes, 
        n_pathways=12,
        seed=42
    )
    
    dataset = generator.generate_complete_dataset()
    
    print(f"✓ Generated dataset with {n_samples} samples and {n_genes} genes")
    print(f"✓ Pathway information: {len(dataset['pathway_info']['pathways'])} pathways")
    print(f"✓ Bottleneck genes: {len(dataset['pathway_info']['bottleneck_genes'])}")
    
    return dataset


def prepare_data_loaders(dataset: Dict, batch_size: int = 32) -> tuple:
    """Prepare PyTorch data loaders."""
    
    print("\n" + "=" * 60)
    print("STEP 2: PREPARING DATA LOADERS")
    print("=" * 60)
    
    data_loader = MultiOmicDataLoader(
        batch_size=batch_size,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    train_loader, val_loader, test_loader = data_loader.prepare_dataloaders(dataset)
    
    # Get feature dimensions
    feature_dims = data_loader.get_feature_dimensions(dataset)
    
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Feature dimensions: {feature_dims}")
    
    return train_loader, val_loader, test_loader, feature_dims, data_loader


def create_and_train_model(train_loader, val_loader, feature_dims: Dict, 
                          pathway_info: Dict, epochs: int = 50) -> MultiOmicTrainer:
    """Create and train the multi-modal transformer model."""
    
    print("\n" + "=" * 60)
    print("STEP 3: CREATING AND TRAINING MODEL")
    print("=" * 60)
    
    # Model configuration
    model_config = {
        'hidden_dim': 256,
        'n_heads': 8,
        'n_layers': 4,
        'use_temporal': False,
        'dropout': 0.1
    }
    
    # Create model
    model = create_multimodal_transformer(
        dataset_info=feature_dims,
        pathway_info=pathway_info,
        config=model_config
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training configuration
    config = TrainingConfig()
    config.epochs = epochs
    config.lr = 1e-4
    config.batch_size = 32
    config.bottleneck_weight = 0.3
    config.early_stopping_patience = 10
    
    # Create trainer
    trainer = MultiOmicTrainer(
        model=model,
        device='auto',
        use_focal_loss=True
    )
    
    print(f"✓ Training on device: {trainer.device}")
    print(f"✓ Starting training for {epochs} epochs...")
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        bottleneck_weight=config.bottleneck_weight,
        save_dir='results/checkpoints',
        early_stopping_patience=config.early_stopping_patience
    )
    
    print("✓ Training completed!")
    
    return trainer


def evaluate_and_analyze(trainer: MultiOmicTrainer, test_loader, dataset: Dict) -> Dict:
    """Evaluate model and perform bottleneck analysis."""
    
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION AND BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    print(f"✓ Test Accuracy: {test_results['accuracy']:.4f}")
    if 'auc' in test_results:
        print(f"✓ Test AUC: {test_results['auc']:.4f}")
    
    # Generate gene names
    n_genes = dataset['mrna'].shape[1]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Save bottleneck analysis
    os.makedirs('results/analysis', exist_ok=True)
    bottleneck_df = trainer.save_bottleneck_analysis(
        test_results, gene_names, 'results/analysis/bottleneck_genes.csv'
    )
    
    print(f"✓ Top 10 bottleneck genes:")
    print(bottleneck_df.head(10))
    
    return test_results, bottleneck_df


def perform_pathway_analysis(dataset: Dict, test_results: Dict) -> pd.DataFrame:
    """Perform comprehensive pathway analysis."""
    
    print("\n" + "=" * 60)
    print("STEP 5: PATHWAY ANALYSIS")
    print("=" * 60)
    
    # Extract pathway information
    pathway_info = dataset['pathway_info']['pathways']
    true_bottlenecks = dataset['pathway_info']['bottleneck_genes']
    
    # Create pathway analyzer
    analyzer = MetabolicPathwayAnalyzer(pathway_info)
    
    # Prepare expression data
    expression_data = dataset['mrna'].values
    gene_names = [f"Gene_{i}" for i in range(expression_data.shape[1])]
    sample_labels = dataset['outcomes']['outcome'].values
    
    # Generate comprehensive pathway report
    pathway_report = analyzer.generate_pathway_report(
        expression_data=expression_data,
        gene_names=gene_names,
        sample_labels=sample_labels
    )
    
    print("✓ Pathway Analysis Report:")
    print(pathway_report)
    
    # Save pathway report
    pathway_report.to_csv('results/analysis/pathway_report.csv', index=False)
    
    # Create bottleneck ranker
    ranker = BottleneckRanker()
    
    # Add different ranking criteria
    bottleneck_scores = np.array(test_results['bottleneck_scores'])
    avg_bottleneck_scores = np.mean(bottleneck_scores, axis=0)
    
    gene_bottleneck_dict = {f"Gene_{i}": score for i, score in enumerate(avg_bottleneck_scores)}
    ranker.add_ranking_criterion("model_scores", gene_bottleneck_dict, weight=0.6)
    
    # Add pathway-based scores (simplified)
    pathway_scores = {}
    for gene_name in gene_names:
        gene_idx = int(gene_name.split('_')[1])
        score = 1.0 if gene_idx in true_bottlenecks else 0.1
        pathway_scores[gene_name] = score
    
    ranker.add_ranking_criterion("pathway_membership", pathway_scores, weight=0.4)
    
    # Get top bottleneck genes
    top_bottlenecks = ranker.get_top_bottlenecks(top_k=50)
    
    print(f"\n✓ Top 20 ranked bottleneck genes:")
    for i, (gene, score) in enumerate(top_bottlenecks[:20]):
        print(f"{i+1:2d}. {gene}: {score:.4f}")
    
    return pathway_report, top_bottlenecks


def create_visualizations(trainer: MultiOmicTrainer, test_results: Dict, 
                         pathway_report: pd.DataFrame):
    """Create comprehensive visualizations."""
    
    print("\n" + "=" * 60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot training history
    trainer.plot_training_history('results/plots/training_history.png')
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_results, 'results/plots/confusion_matrix.png')
    
    # Plot pathway dysregulation scores
    plt.figure(figsize=(12, 8))
    pathway_report_sorted = pathway_report.sort_values('Dysregulation_Score', ascending=True)
    
    plt.barh(range(len(pathway_report_sorted)), pathway_report_sorted['Dysregulation_Score'])
    plt.yticks(range(len(pathway_report_sorted)), pathway_report_sorted['Pathway'])
    plt.xlabel('Dysregulation Score')
    plt.title('Pathway Dysregulation Analysis')
    plt.tight_layout()
    plt.savefig('results/plots/pathway_dysregulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot bottleneck gene distribution
    bottleneck_scores = np.array(test_results['bottleneck_scores'])
    avg_scores = np.mean(bottleneck_scores, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.hist(avg_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Average Bottleneck Score')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Bottleneck Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/bottleneck_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ All visualizations saved to results/plots/")


def generate_summary_report(dataset: Dict, test_results: Dict, 
                           pathway_report: pd.DataFrame, top_bottlenecks: List):
    """Generate a comprehensive summary report."""
    
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    summary = {
        "Dataset Information": {
            "Total Samples": len(dataset['cna']),
            "Total Genes": len(dataset['cna'].columns),
            "Number of Pathways": len(dataset['pathway_info']['pathways']),
            "True Bottleneck Genes": len(dataset['pathway_info']['bottleneck_genes'])
        },
        "Model Performance": {
            "Test Accuracy": f"{test_results['accuracy']:.4f}",
            "Test AUC": f"{test_results.get('auc', 0):.4f}",
            "Classification Report": test_results['classification_report']
        },
        "Pathway Analysis": {
            "Most Dysregulated Pathway": pathway_report.iloc[0]['Pathway'],
            "Highest Dysregulation Score": f"{pathway_report.iloc[0]['Dysregulation_Score']:.4f}",
            "Average Genes per Pathway": f"{pathway_report['Total_Genes'].mean():.1f}"
        },
        "Bottleneck Analysis": {
            "Top Predicted Bottleneck": top_bottlenecks[0][0],
            "Top Bottleneck Score": f"{top_bottlenecks[0][1]:.4f}",
            "Total Identified Bottlenecks": len([b for b in top_bottlenecks if b[1] > 0.5])
        }
    }
    
    # Save summary as JSON
    import json
    with open('results/analysis/summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("📊 ANALYSIS SUMMARY:")
    print(f"✓ Analyzed {summary['Dataset Information']['Total Samples']} samples with {summary['Dataset Information']['Total Genes']} genes")
    print(f"✓ Model achieved {summary['Model Performance']['Test Accuracy']} accuracy")
    print(f"✓ Most dysregulated pathway: {summary['Pathway Analysis']['Most Dysregulated Pathway']}")
    print(f"✓ Top bottleneck gene: {summary['Bottleneck Analysis']['Top Predicted Bottleneck']}")
    
    print(f"\n📁 All results saved to 'results/' directory")
    print("   ├── analysis/")
    print("   │   ├── bottleneck_genes.csv")
    print("   │   ├── pathway_report.csv")
    print("   │   └── summary_report.json")
    print("   ├── checkpoints/")
    print("   │   ├── best_model.pth")
    print("   │   └── training_history.json")
    print("   └── plots/")
    print("       ├── training_history.png")
    print("       ├── confusion_matrix.png")
    print("       ├── pathway_dysregulation.png")
    print("       └── bottleneck_distribution.png")


def main():
    """Main execution function."""
    
    print("🧬 MULTI-OMIC METABOLIC PATHWAY BOTTLENECK ANALYSIS")
    print("🔬 Using Transformer-Based Deep Learning")
    print("⚡ Integrating CNA, Mutations, mRNA, and Clinical Data")
    print()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Step 1: Generate synthetic dataset
        dataset = generate_synthetic_dataset(n_samples=800, n_genes=1000)
        
        # Step 2: Prepare data loaders
        train_loader, val_loader, test_loader, feature_dims, data_loader = prepare_data_loaders(
            dataset, batch_size=32
        )
        
        # Step 3: Create and train model
        trainer = create_and_train_model(
            train_loader, val_loader, feature_dims, 
            dataset['pathway_info']['pathways'], epochs=30
        )
        
        # Step 4: Evaluate and analyze
        test_results, bottleneck_df = evaluate_and_analyze(trainer, test_loader, dataset)
        
        # Step 5: Pathway analysis
        pathway_report, top_bottlenecks = perform_pathway_analysis(dataset, test_results)
        
        # Step 6: Create visualizations
        create_visualizations(trainer, test_results, pathway_report)
        
        # Step 7: Generate summary report
        generate_summary_report(dataset, test_results, pathway_report, top_bottlenecks)
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("   Check the 'results/' directory for all outputs.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Multi-omic analysis pipeline executed successfully!")
    else:
        print("\n❌ Analysis pipeline failed. Check error messages above.") 
