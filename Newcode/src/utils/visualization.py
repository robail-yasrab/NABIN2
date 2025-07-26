import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import networkx as nx
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_multi_omic_overview(dataset: Dict, save_path: Optional[str] = None):
    """Create overview plots of multi-omic data."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # CNA distribution
    axes[0, 0].hist(dataset['cna'].values.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Copy Number Alterations Distribution')
    axes[0, 0].set_xlabel('CNA Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Mutation frequency
    mutation_freq = dataset['mutations'].sum(axis=0)
    axes[0, 1].hist(mutation_freq, bins=30, alpha=0.7, color='red')
    axes[0, 1].set_title('Mutation Frequency per Gene')
    axes[0, 1].set_xlabel('Number of Mutations')
    axes[0, 1].set_ylabel('Number of Genes')
    
    # mRNA expression
    axes[0, 2].hist(np.log2(dataset['mrna'].values.flatten() + 1), bins=50, alpha=0.7, color='green')
    axes[0, 2].set_title('mRNA Expression Distribution (log2)')
    axes[0, 2].set_xlabel('Log2(Expression + 1)')
    axes[0, 2].set_ylabel('Frequency')
    
    # Clinical features correlation
    clinical_data = dataset['clinical'][dataset['clinical'].index.str.endswith('_T0')]
    corr_matrix = clinical_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Clinical Features Correlation')
    
    # Outcome distribution
    outcome_counts = dataset['outcomes']['outcome'].value_counts()
    axes[1, 1].pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Outcome Distribution')
    
    # Sample correlation heatmap (subset)
    sample_subset = dataset['mrna'].iloc[:50, :50]  # First 50x50 for visualization
    corr_subset = sample_subset.corr()
    sns.heatmap(corr_subset, cmap='viridis', ax=axes[1, 2], cbar=False)
    axes[1, 2].set_title('Gene Expression Correlation (subset)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pathway_network_interactive(pathway_info: Dict, bottleneck_scores: Dict, 
                                   correlation_matrix: np.ndarray, gene_names: List[str],
                                   pathway_name: str = None):
    """Create interactive pathway network visualization."""
    
    if pathway_name and pathway_name in pathway_info:
        pathways_to_plot = {pathway_name: pathway_info[pathway_name]}
    else:
        pathways_to_plot = pathway_info
    
    fig = make_subplots(
        rows=1, cols=len(pathways_to_plot),
        subplot_titles=list(pathways_to_plot.keys()),
        specs=[[{"type": "scatter"}] * len(pathways_to_plot)]
    )
    
    for idx, (pathway, gene_indices) in enumerate(pathways_to_plot.items()):
        if len(gene_indices) == 0:
            continue
            
        # Create subgraph
        valid_indices = [i for i in gene_indices if i < len(gene_names)]
        pathway_genes = [gene_names[i] for i in valid_indices]
        
        # Create network
        G = nx.Graph()
        G.add_nodes_from(pathway_genes)
        
        # Add edges based on correlation
        threshold = 0.3
        for i, gene1 in enumerate(pathway_genes):
            for j, gene2 in enumerate(pathway_genes):
                if i < j:
                    idx1, idx2 = valid_indices[i], valid_indices[j]
                    if abs(correlation_matrix[idx1, idx2]) > threshold:
                        G.add_edge(gene1, gene2, weight=abs(correlation_matrix[idx1, idx2]))
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [bottleneck_scores.get(node, 0) for node in G.nodes()]
        
        # Edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Add edge trace
        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'),
                      showlegend=False, hoverinfo='none'),
            row=1, col=idx+1
        )
        
        # Add node trace
        fig.add_trace(
            go.Scatter(x=node_x, y=node_y, mode='markers+text',
                      marker=dict(size=10, color=node_colors, colorscale='Reds',
                                showscale=True if idx == 0 else False),
                      text=list(G.nodes()), textposition="middle center",
                      hoverinfo='text', showlegend=False),
            row=1, col=idx+1
        )
    
    fig.update_layout(height=600, showlegend=False, title="Pathway Networks with Bottleneck Scores")
    fig.show()


def plot_attention_heatmap(attention_weights: torch.Tensor, gene_names: List[str], 
                          top_k: int = 50, save_path: Optional[str] = None):
    """Plot attention weights heatmap."""
    
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Average across samples and heads if multi-dimensional
    if attention_weights.ndim > 2:
        attention_weights = np.mean(attention_weights, axis=(0, 1))
    elif attention_weights.ndim > 1:
        attention_weights = np.mean(attention_weights, axis=0)
    
    # Select top genes by attention
    top_indices = np.argsort(attention_weights)[-top_k:]
    top_genes = [gene_names[i] for i in top_indices if i < len(gene_names)]
    top_attention = attention_weights[top_indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_genes)), top_attention)
    plt.yticks(range(len(top_genes)), top_genes)
    plt.xlabel('Attention Weight')
    plt.title(f'Top {top_k} Genes by Attention Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_bottleneck_ranking(bottleneck_scores: np.ndarray, gene_names: List[str],
                           true_bottlenecks: List[int] = None, top_k: int = 50,
                           save_path: Optional[str] = None):
    """Plot bottleneck gene ranking with true positives highlighted."""
    
    # Average scores across samples
    if bottleneck_scores.ndim > 1:
        avg_scores = np.mean(bottleneck_scores, axis=0)
    else:
        avg_scores = bottleneck_scores
    
    # Get top genes
    top_indices = np.argsort(avg_scores)[-top_k:][::-1]
    top_genes = [gene_names[i] for i in top_indices if i < len(gene_names)]
    top_scores = avg_scores[top_indices]
    
    # Color code based on true bottlenecks
    colors = []
    if true_bottlenecks:
        for idx in top_indices:
            if idx in true_bottlenecks:
                colors.append('red')  # True bottleneck
            else:
                colors.append('blue')  # Predicted bottleneck
    else:
        colors = ['blue'] * len(top_genes)
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(top_genes)), top_scores, color=colors, alpha=0.7)
    plt.yticks(range(len(top_genes)), top_genes)
    plt.xlabel('Bottleneck Score')
    plt.title(f'Top {top_k} Predicted Bottleneck Genes')
    
    if true_bottlenecks:
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='True Bottleneck'),
                          Patch(facecolor='blue', label='Predicted Only')]
        plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pathway_importance_radar(pathway_importance: Dict[str, float], 
                                save_path: Optional[str] = None):
    """Create radar plot of pathway importance scores."""
    
    pathways = list(pathway_importance.keys())
    scores = list(pathway_importance.values())
    
    # Normalize scores to 0-1 range
    max_score = max(scores)
    normalized_scores = [s / max_score for s in scores]
    
    # Add first value to end to close the radar chart
    pathways_plot = pathways + [pathways[0]]
    scores_plot = normalized_scores + [normalized_scores[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores_plot,
        theta=pathways_plot,
        fill='toself',
        name='Pathway Importance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Pathway Importance Radar Chart"
    )
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
    fig.show()


def plot_temporal_clinical_trends(clinical_data: pd.DataFrame, 
                                save_path: Optional[str] = None):
    """Plot temporal trends in clinical data."""
    
    # Extract timepoint data
    temporal_data = {}
    for idx in clinical_data.index:
        if '_T' in idx:
            sample_id = idx.split('_T')[0]
            timepoint = int(idx.split('_T')[1])
            if sample_id not in temporal_data:
                temporal_data[sample_id] = {}
            temporal_data[sample_id][timepoint] = clinical_data.loc[idx]
    
    # Create plots for key clinical features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    clinical_features = ['tumor_size', 'survival_months', 'treatment_response', 'metastasis']
    
    for idx, feature in enumerate(clinical_features):
        ax = axes[idx // 2, idx % 2]
        
        # Aggregate data across samples
        timepoints = sorted(set().union(*[list(data.keys()) for data in temporal_data.values()]))
        mean_values = []
        std_values = []
        
        for t in timepoints:
            values = []
            for sample_data in temporal_data.values():
                if t in sample_data:
                    values.append(sample_data[t][feature])
            
            if values:
                mean_values.append(np.mean(values))
                std_values.append(np.std(values))
            else:
                mean_values.append(0)
                std_values.append(0)
        
        ax.errorbar(timepoints, mean_values, yerr=std_values, marker='o', capsize=5)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()} Over Time')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_dashboard(dataset: Dict, model_results: Dict, 
                                 pathway_analysis: Dict, save_dir: str = 'dashboard'):
    """Create a comprehensive analysis dashboard."""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating comprehensive analysis dashboard...")
    
    # 1. Multi-omic overview
    plot_multi_omic_overview(dataset, f"{save_dir}/multiomics_overview.png")
    
    # 2. Model performance
    if 'history' in model_results:
        plot_training_curves(model_results['history'], f"{save_dir}/training_curves.png")
    
    # 3. Bottleneck analysis
    if 'bottleneck_scores' in model_results:
        plot_bottleneck_ranking(
            model_results['bottleneck_scores'],
            [f"Gene_{i}" for i in range(dataset['mrna'].shape[1])],
            dataset['pathway_info']['bottleneck_genes'],
            save_path=f"{save_dir}/bottleneck_ranking.png"
        )
    
    # 4. Pathway importance
    if 'pathway_importance' in pathway_analysis:
        plot_pathway_importance_radar(
            pathway_analysis['pathway_importance'],
            f"{save_dir}/pathway_radar.png"
        )
    
    # 5. Temporal clinical trends
    plot_temporal_clinical_trends(dataset['clinical'], f"{save_dir}/clinical_trends.png")
    
    print(f"Dashboard saved to {save_dir}/ directory")


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """Plot training curves from history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classification loss
    axes[1, 0].plot(history['train_cls_loss'], label='Train Classification Loss')
    axes[1, 0].plot(history['val_cls_loss'], label='Val Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Bottleneck loss
    axes[1, 1].plot(history['train_btl_loss'], label='Train Bottleneck Loss')
    axes[1, 1].plot(history['val_btl_loss'], label='Val Bottleneck Loss')
    axes[1, 1].set_title('Bottleneck Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("- plot_multi_omic_overview()")
    print("- plot_pathway_network_interactive()")
    print("- plot_attention_heatmap()")
    print("- plot_bottleneck_ranking()")
    print("- plot_pathway_importance_radar()")
    print("- plot_temporal_clinical_trends()")
    print("- create_comprehensive_dashboard()") 