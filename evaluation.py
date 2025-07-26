import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from multimodal_models import MultimodalVAE
from data_generator import MultiOmicsDataGenerator

class MultiOmicsEvaluator:
    """Comprehensive evaluation toolkit for multimodal multi-omics analysis."""
    
    def __init__(self, model: MultimodalVAE, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_reconstruction_quality(self, dataset: Dict) -> Dict[str, Dict[str, float]]:
        """Evaluate reconstruction quality for each modality."""
        with torch.no_grad():
            # Prepare data
            data = {
                'mrna_expression': dataset['mrna_expression'].to(self.device),
                'cna': dataset['cna'].to(self.device),
                'mutations': dataset['mutations'].to(self.device)
            }
            
            # Forward pass
            output = self.model(data)
            reconstructions = output['reconstructions']
            
            # Compute reconstruction metrics
            metrics = {}
            
            # mRNA reconstruction metrics
            mrna_true = data['mrna_expression'].cpu().numpy()
            mrna_recon = reconstructions['mrna_recon'].cpu().numpy()
            
            metrics['mrna'] = {
                'mse': mean_squared_error(mrna_true, mrna_recon),
                'mae': mean_absolute_error(mrna_true, mrna_recon),
                'r2': r2_score(mrna_true.flatten(), mrna_recon.flatten()),
                'pearson_r': pearsonr(mrna_true.flatten(), mrna_recon.flatten())[0]
            }
            
            # CNA reconstruction metrics
            cna_true = data['cna'].cpu().numpy()
            cna_recon = reconstructions['cna_recon'].cpu().numpy()
            
            metrics['cna'] = {
                'mse': mean_squared_error(cna_true, cna_recon),
                'mae': mean_absolute_error(cna_true, cna_recon),
                'pearson_r': pearsonr(cna_true.flatten(), cna_recon.flatten())[0]
            }
            
            # Mutation reconstruction metrics
            mutations_true = data['mutations'].cpu().numpy()
            mutations_recon = reconstructions['mutations_recon'].cpu().numpy()
            
            # Convert to binary predictions
            mutations_pred_binary = (mutations_recon > 0.5).astype(int)
            
            metrics['mutations'] = {
                'accuracy': np.mean(mutations_true == mutations_pred_binary),
                'precision': self._safe_precision(mutations_true.flatten(), mutations_pred_binary.flatten()),
                'recall': self._safe_recall(mutations_true.flatten(), mutations_pred_binary.flatten()),
                'auc': roc_curve(mutations_true.flatten(), mutations_recon.flatten())[2] if len(np.unique(mutations_true)) > 1 else 0.5
            }
            
        return metrics
    
    def evaluate_bottleneck_identification(self, dataset: Dict) -> Dict[str, float]:
        """Evaluate bottleneck gene identification performance."""
        with torch.no_grad():
            # Prepare data
            data = {
                'mrna_expression': dataset['mrna_expression'].to(self.device),
                'cna': dataset['cna'].to(self.device),
                'mutations': dataset['mutations'].to(self.device)
            }
            
            # Forward pass
            output = self.model(data)
            bottleneck_scores = output['bottleneck_analysis']['bottleneck_scores'].cpu().numpy()
            
            # Create ground truth labels
            n_samples, n_genes = bottleneck_scores.shape
            true_bottleneck_genes = dataset['bottleneck_genes']
            
            y_true = np.zeros((n_samples, n_genes))
            for gene_idx in true_bottleneck_genes:
                y_true[:, gene_idx] = 1
            
            # Compute metrics
            y_true_flat = y_true.flatten()
            y_scores_flat = bottleneck_scores.flatten()
            
            # ROC-AUC
            fpr, tpr, _ = roc_curve(y_true_flat, y_scores_flat)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_true_flat, y_scores_flat)
            pr_auc = auc(recall, precision)
            
            # Top-k accuracy (percentage of true bottlenecks in top predictions)
            k_values = [10, 20, 50, 100]
            top_k_accuracies = {}
            
            for k in k_values:
                # Get top-k predictions for each sample
                top_k_correct = 0
                total_predictions = 0
                
                for sample_idx in range(n_samples):
                    sample_scores = bottleneck_scores[sample_idx]
                    sample_true = y_true[sample_idx]
                    
                    # Get top-k predicted genes
                    top_k_genes = np.argsort(sample_scores)[-k:]
                    
                    # Count how many are actually bottlenecks
                    correct_predictions = np.sum(sample_true[top_k_genes])
                    top_k_correct += correct_predictions
                    total_predictions += k
                
                top_k_accuracies[f'top_{k}_accuracy'] = top_k_correct / total_predictions
            
            metrics = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                **top_k_accuracies
            }
            
        return metrics
    
    def evaluate_clinical_predictions(self, dataset: Dict) -> Dict[str, Dict[str, float]]:
        """Evaluate clinical outcome predictions."""
        with torch.no_grad():
            # Prepare data
            data = {
                'mrna_expression': dataset['mrna_expression'].to(self.device),
                'cna': dataset['cna'].to(self.device),
                'mutations': dataset['mutations'].to(self.device)
            }
            
            # Forward pass
            output = self.model(data)
            clinical_predictions = output['clinical_predictions']
            
            metrics = {}
            
            # Survival prediction metrics
            if 'survival_pred' in clinical_predictions:
                survival_true = dataset['overall_survival'].numpy()
                survival_pred = clinical_predictions['survival_pred'].cpu().numpy()
                
                metrics['survival'] = {
                    'mse': mean_squared_error(survival_true, survival_pred),
                    'mae': mean_absolute_error(survival_true, survival_pred),
                    'r2': r2_score(survival_true, survival_pred),
                    'pearson_r': pearsonr(survival_true, survival_pred)[0]
                }
            
            # Response prediction metrics
            if 'response_pred' in clinical_predictions:
                response_true = dataset['response_score'].numpy()
                response_pred = clinical_predictions['response_pred'].cpu().numpy()
                
                metrics['response'] = {
                    'mse': mean_squared_error(response_true, response_pred),
                    'mae': mean_absolute_error(response_true, response_pred),
                    'r2': r2_score(response_true, response_pred),
                    'pearson_r': pearsonr(response_true, response_pred)[0]
                }
            
        return metrics
    
    def analyze_latent_space(self, dataset: Dict) -> Dict[str, np.ndarray]:
        """Analyze the learned latent space representations."""
        with torch.no_grad():
            # Prepare data
            data = {
                'mrna_expression': dataset['mrna_expression'].to(self.device),
                'cna': dataset['cna'].to(self.device),
                'mutations': dataset['mutations'].to(self.device)
            }
            
            # Forward pass
            output = self.model(data)
            latent_representations = output['latent_representations']
            
            # Extract latent representations
            analysis = {}
            
            # Fused latent space
            analysis['fused_latent'] = latent_representations['fused_z'].cpu().numpy()
            
            # Individual modality latent spaces
            for i, modality in enumerate(['mrna', 'cna', 'mutations']):
                analysis[f'{modality}_latent'] = latent_representations['modality_zs'][i].cpu().numpy()
            
        return analysis
    
    def _safe_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute precision with zero division handling."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _safe_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute recall with zero division handling."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

class BottleneckVisualizer:
    """Visualization tools for bottleneck gene analysis."""
    
    def __init__(self, model: MultimodalVAE, dataset: Dict, device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.evaluator = MultiOmicsEvaluator(model, device)
    
    def plot_bottleneck_scores(self, save_path: str = None):
        """Plot bottleneck gene scores."""
        with torch.no_grad():
            # Get bottleneck scores
            data = {
                'mrna_expression': self.dataset['mrna_expression'].to(self.device),
                'cna': self.dataset['cna'].to(self.device),
                'mutations': self.dataset['mutations'].to(self.device)
            }
            
            output = self.model(data)
            bottleneck_scores = output['bottleneck_analysis']['bottleneck_scores'].cpu().numpy()
            
            # Average scores across samples
            avg_scores = np.mean(bottleneck_scores, axis=0)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Histogram of bottleneck scores
            axes[0, 0].hist(avg_scores, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of Average Bottleneck Scores')
            axes[0, 0].set_xlabel('Bottleneck Score')
            axes[0, 0].set_ylabel('Number of Genes')
            
            # Highlight true bottlenecks
            true_bottleneck_scores = avg_scores[self.dataset['bottleneck_genes']]
            axes[0, 0].axvline(np.mean(true_bottleneck_scores), color='red', linestyle='--', 
                              label=f'True Bottlenecks Mean: {np.mean(true_bottleneck_scores):.3f}')
            axes[0, 0].legend()
            
            # Plot 2: Top predicted bottlenecks
            top_genes = np.argsort(avg_scores)[-20:]
            top_scores = avg_scores[top_genes]
            gene_names = [self.dataset['gene_names'][i] for i in top_genes]
            
            colors = ['red' if i in self.dataset['bottleneck_genes'] else 'blue' for i in top_genes]
            
            axes[0, 1].barh(range(len(top_genes)), top_scores, color=colors, alpha=0.7)
            axes[0, 1].set_yticks(range(len(top_genes)))
            axes[0, 1].set_yticklabels(gene_names, fontsize=8)
            axes[0, 1].set_title('Top 20 Predicted Bottleneck Genes')
            axes[0, 1].set_xlabel('Bottleneck Score')
            
            # Add legend
            red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='True Bottleneck')
            blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='False Positive')
            axes[0, 1].legend(handles=[red_patch, blue_patch])
            
            # Plot 3: ROC curve
            y_true = np.zeros(len(avg_scores))
            y_true[self.dataset['bottleneck_genes']] = 1
            
            fpr, tpr, _ = roc_curve(y_true, avg_scores)
            roc_auc = auc(fpr, tpr)
            
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve for Bottleneck Gene Identification')
            axes[1, 0].legend(loc="lower right")
            
            # Plot 4: Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, avg_scores)
            pr_auc = auc(recall, precision)
            
            axes[1, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].legend(loc="lower left")
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_pathway_analysis(self, save_path: str = None):
        """Visualize pathway structure and bottleneck gene locations."""
        # Get pathway graph
        G = self.dataset['pathway_graph']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Pathway network
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node colors based on bottleneck status
        node_colors = []
        for node in G.nodes():
            if node in self.dataset['bottleneck_genes']:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        nx.draw(G, pos, ax=axes[0], node_color=node_colors, node_size=50, 
                with_labels=False, edge_color='gray', alpha=0.6)
        axes[0].set_title('Metabolic Pathway Network\n(Red: Bottleneck Genes, Blue: Regular Genes)')
        
        # Plot 2: Pathway-wise bottleneck distribution
        pathway_bottleneck_counts = {}
        for pathway, genes in self.dataset['pathway_genes'].items():
            bottleneck_count = sum(1 for gene in genes if gene in self.dataset['bottleneck_genes'])
            pathway_bottleneck_counts[pathway] = bottleneck_count
        
        pathways = list(pathway_bottleneck_counts.keys())
        counts = list(pathway_bottleneck_counts.values())
        
        axes[1].bar(range(len(pathways)), counts, alpha=0.7, color='orange')
        axes[1].set_xticks(range(len(pathways)))
        axes[1].set_xticklabels(pathways, rotation=45, ha='right')
        axes[1].set_title('Bottleneck Genes per Pathway')
        axes[1].set_ylabel('Number of Bottleneck Genes')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clinical_predictions(self, save_path: str = None):
        """Visualize clinical outcome predictions."""
        with torch.no_grad():
            # Get predictions
            data = {
                'mrna_expression': self.dataset['mrna_expression'].to(self.device),
                'cna': self.dataset['cna'].to(self.device),
                'mutations': self.dataset['mutations'].to(self.device)
            }
            
            output = self.model(data)
            clinical_predictions = output['clinical_predictions']
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Survival prediction
            if 'survival_pred' in clinical_predictions:
                survival_true = self.dataset['overall_survival'].numpy()
                survival_pred = clinical_predictions['survival_pred'].cpu().numpy()
                
                axes[0].scatter(survival_true, survival_pred, alpha=0.6)
                
                # Perfect prediction line
                min_val = min(survival_true.min(), survival_pred.min())
                max_val = max(survival_true.max(), survival_pred.max())
                axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                # Calculate R²
                r2 = r2_score(survival_true, survival_pred)
                axes[0].set_title(f'Survival Prediction (R² = {r2:.3f})')
                axes[0].set_xlabel('True Overall Survival (months)')
                axes[0].set_ylabel('Predicted Overall Survival (months)')
            
            # Response prediction
            if 'response_pred' in clinical_predictions:
                response_true = self.dataset['response_score'].numpy()
                response_pred = clinical_predictions['response_pred'].cpu().numpy()
                
                axes[1].scatter(response_true, response_pred, alpha=0.6, color='orange')
                
                # Perfect prediction line
                axes[1].plot([0, 100], [0, 100], 'r--', lw=2)
                
                # Calculate R²
                r2 = r2_score(response_true, response_pred)
                axes[1].set_title(f'Response Score Prediction (R² = {r2:.3f})')
                axes[1].set_xlabel('True Response Score')
                axes[1].set_ylabel('Predicted Response Score')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_comprehensive_report(self, save_dir: str = './results/'):
        """Create a comprehensive evaluation report."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive evaluation report...")
        
        # Evaluate all components
        reconstruction_metrics = self.evaluator.evaluate_reconstruction_quality(self.dataset)
        bottleneck_metrics = self.evaluator.evaluate_bottleneck_identification(self.dataset)
        clinical_metrics = self.evaluator.evaluate_clinical_predictions(self.dataset)
        
        # Generate visualizations
        self.plot_bottleneck_scores(os.path.join(save_dir, 'bottleneck_analysis.png'))
        self.plot_pathway_analysis(os.path.join(save_dir, 'pathway_analysis.png'))
        self.plot_clinical_predictions(os.path.join(save_dir, 'clinical_predictions.png'))
        
        # Create summary report
        report = {
            'Reconstruction Quality': reconstruction_metrics,
            'Bottleneck Identification': bottleneck_metrics,
            'Clinical Predictions': clinical_metrics
        }
        
        # Save metrics to file
        import json
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            # Convert numpy values to regular Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump(convert_numpy(report), f, indent=2)
        
        print(f"Evaluation report saved to {save_dir}")
        return report

if __name__ == "__main__":
    # Load or generate dataset
    print("Loading dataset...")
    try:
        dataset = torch.load('multiomics_dataset.pt')
    except FileNotFoundError:
        print("Generating new dataset...")
        generator = MultiOmicsDataGenerator(n_samples=1000, n_genes=500, n_pathways=20)
        dataset = generator.generate_complete_dataset()
        torch.save(dataset, 'multiomics_dataset.pt')
    
    # Load trained model (assuming it exists)
    try:
        from multimodal_models import create_model_config
        config = create_model_config(n_genes=500)
        model = MultimodalVAE(config)
        
        checkpoint = torch.load('multiomics_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Model loaded successfully!")
        
        # Create visualizer and generate report
        visualizer = BottleneckVisualizer(model, dataset)
        report = visualizer.create_comprehensive_report()
        
        print("Evaluation completed!")
        
    except FileNotFoundError:
        print("Trained model not found. Please run training.py first.") 