#!/usr/bin/env python3
"""
Main execution script for Multi-Omics Metabolic Pathway Bottleneck Analysis

This script orchestrates the complete pipeline:
1. Data generation (synthetic multi-omics data)
2. Model training (Multimodal VAE with Graph Neural Networks)
3. Evaluation and visualization
4. Bottleneck gene identification and analysis

Author: Robail Yasrab
Date: 2025
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_generator import MultiOmicsDataGenerator
from multimodal_models import MultimodalVAE, create_model_config
from training import MultiOmicsTrainer, plot_training_history
from evaluation import BottleneckVisualizer, MultiOmicsEvaluator

def main():
    parser = argparse.ArgumentParser(description='Multi-Omics Metabolic Pathway Bottleneck Analysis')
    
    # Data parameters
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of patient samples')
    parser.add_argument('--n_genes', type=int, default=500, help='Number of genes')
    parser.add_argument('--n_pathways', type=int, default=20, help='Number of metabolic pathways')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent space dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL divergence loss weight')
    parser.add_argument('--clinical_weight', type=float, default=0.5, help='Clinical prediction loss weight')
    parser.add_argument('--bottleneck_weight', type=float, default=0.3, help='Bottleneck identification loss weight')
    parser.add_argument('--pathway_weight', type=float, default=0.2, help='Pathway structure loss weight')
    
    # Execution options
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'data_only', 'train_only', 'eval_only'],
                       help='Execution mode')
    parser.add_argument('--output_dir', type=str, default='./results/', help='Output directory')
    parser.add_argument('--load_data', type=str, default=None, help='Path to load existing dataset')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load existing model')
    parser.add_argument('--save_model', type=str, default='multiomics_model.pth', help='Path to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Data Generation/Loading
    if args.mode in ['full', 'data_only']:
        print("="*60)
        print("STEP 1: DATA GENERATION")
        print("="*60)
        
        if args.load_data and os.path.exists(args.load_data):
            print(f"Loading existing dataset from {args.load_data}")
            dataset = torch.load(args.load_data)
        else:
            print("Generating synthetic multi-omics dataset...")
            generator = MultiOmicsDataGenerator(
                n_samples=args.n_samples,
                n_genes=args.n_genes,
                n_pathways=args.n_pathways,
                seed=args.seed
            )
            dataset = generator.generate_complete_dataset()
            
            # Save dataset
            dataset_path = os.path.join(args.output_dir, 'multiomics_dataset.pt')
            torch.save(dataset, dataset_path)
            print(f"Dataset saved to {dataset_path}")
            
            # Generate data summary visualization
            generator.visualize_data_summary(
                dataset, 
                os.path.join(args.output_dir, 'data_summary.png')
            )
        
        print(f"Dataset Summary:")
        print(f"  - Samples: {dataset['mrna_expression'].shape[0]}")
        print(f"  - Genes: {dataset['mrna_expression'].shape[1]}")
        print(f"  - Bottleneck genes: {len(dataset['bottleneck_genes'])}")
        print(f"  - Pathways: {len(dataset['pathway_genes'])}")
        
        if args.mode == 'data_only':
            return
    
    else:
        # Load existing dataset
        if args.load_data and os.path.exists(args.load_data):
            dataset = torch.load(args.load_data)
        else:
            dataset_path = os.path.join(args.output_dir, 'multiomics_dataset.pt')
            if os.path.exists(dataset_path):
                dataset = torch.load(dataset_path)
            else:
                raise FileNotFoundError("No dataset found. Please run with 'full' or 'data_only' mode first.")
    
    # Step 2: Model Training
    if args.mode in ['full', 'train_only']:
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)
        
        # Create model configuration
        model_config = create_model_config(n_genes=args.n_genes)
        model_config.update({
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': 1e-4,
            'patience': 20,
            'recon_weight': args.recon_weight,
            'kl_weight': args.kl_weight,
            'clinical_weight': args.clinical_weight,
            'bottleneck_weight': args.bottleneck_weight,
            'pathway_weight': args.pathway_weight
        })
        
        # Initialize model and trainer
        model = MultimodalVAE(model_config)
        trainer = MultiOmicsTrainer(model, model_config)
        
        # Train the model
        print("Starting training...")
        training_results = trainer.train(dataset, num_epochs=args.num_epochs)
        
        # Save model
        model_path = os.path.join(args.output_dir, args.save_model)
        trainer.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot training history
        plot_training_history(
            training_results['train_history'],
            training_results['val_history'],
            os.path.join(args.output_dir, 'training_history.png')
        )
        
        print(f"Training completed! Best validation loss: {training_results['best_val_loss']:.4f}")
        
        if args.mode == 'train_only':
            return
    
    else:
        # Load existing model
        if args.load_model and os.path.exists(args.load_model):
            model_path = args.load_model
        else:
            model_path = os.path.join(args.output_dir, args.save_model)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found. Please run training first.")
        
        # Load model
        model_config = create_model_config(n_genes=args.n_genes)
        model = MultimodalVAE(model_config)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    
    # Step 3: Evaluation and Analysis
    print("\n" + "="*60)
    print("STEP 3: EVALUATION AND ANALYSIS")
    print("="*60)
    
    # Create evaluator and visualizer
    evaluator = MultiOmicsEvaluator(model, device=str(device))
    visualizer = BottleneckVisualizer(model, dataset, device=str(device))
    
    # Comprehensive evaluation
    print("Evaluating model performance...")
    
    # Reconstruction quality
    reconstruction_metrics = evaluator.evaluate_reconstruction_quality(dataset)
    print("\nReconstruction Quality:")
    for modality, metrics in reconstruction_metrics.items():
        print(f"  {modality.upper()}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Bottleneck identification
    bottleneck_metrics = evaluator.evaluate_bottleneck_identification(dataset)
    print("\nBottleneck Gene Identification:")
    for metric, value in bottleneck_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Clinical predictions
    clinical_metrics = evaluator.evaluate_clinical_predictions(dataset)
    print("\nClinical Outcome Predictions:")
    for outcome, metrics in clinical_metrics.items():
        print(f"  {outcome.upper()}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Generate comprehensive report with visualizations
    print("\nGenerating comprehensive evaluation report...")
    report = visualizer.create_comprehensive_report(args.output_dir)
    
    # Identify top bottleneck genes
    print("\n" + "="*60)
    print("BOTTLENECK GENE ANALYSIS")
    print("="*60)
    
    with torch.no_grad():
        data = {
            'mrna_expression': dataset['mrna_expression'].to(device),
            'cna': dataset['cna'].to(device),
            'mutations': dataset['mutations'].to(device)
        }
        
        output = model(data)
        bottleneck_scores = output['bottleneck_analysis']['bottleneck_scores'].cpu().numpy()
        avg_scores = np.mean(bottleneck_scores, axis=0)
        
        # Get top predicted bottleneck genes
        top_k = 20
        top_gene_indices = np.argsort(avg_scores)[-top_k:][::-1]
        
        print(f"\nTop {top_k} Predicted Bottleneck Genes:")
        print("-" * 50)
        for i, gene_idx in enumerate(top_gene_indices):
            gene_name = dataset['gene_names'][gene_idx]
            score = avg_scores[gene_idx]
            is_true_bottleneck = gene_idx in dataset['bottleneck_genes']
            status = "✓ TRUE" if is_true_bottleneck else "✗ FALSE"
            print(f"{i+1:2d}. {gene_name}: {score:.4f} ({status})")
        
        # Calculate precision at different k values
        print(f"\nPrecision at different K values:")
        print("-" * 30)
        for k in [5, 10, 15, 20]:
            top_k_genes = np.argsort(avg_scores)[-k:]
            true_positives = sum(1 for gene in top_k_genes if gene in dataset['bottleneck_genes'])
            precision = true_positives / k
            print(f"Precision@{k:2d}: {precision:.3f} ({true_positives}/{k})")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")
    print(f"📊 Key files generated:")
    print(f"   - Dataset: multiomics_dataset.pt")
    print(f"   - Model: {args.save_model}")
    print(f"   - Visualizations: *.png files")
    print(f"   - Metrics: evaluation_metrics.json")

def print_banner():
    """Print a fancy banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     Multi-Omics Metabolic Pathway Bottleneck Gene Analysis      ║
    ║                                                                  ║
    ║  🧬 Integrates: CNA + Mutations + mRNA Expression + Clinical     ║
    ║  🤖 AI Method: Multimodal VAE + Graph Neural Networks           ║
    ║  🎯 Goal: Identify bottleneck genes in metabolic pathways       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    print_banner()
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc() 
