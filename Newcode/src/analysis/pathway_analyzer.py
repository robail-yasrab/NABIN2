import torch
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


class MetabolicPathwayAnalyzer:
    """Analyzer for identifying bottleneck genes in metabolic pathways."""
    
    def __init__(self, pathway_info: Dict[str, List[int]]):
        self.pathway_info = pathway_info
        self.pathway_graphs = {}
        self.bottleneck_scores = {}
        self.importance_scores = {}
        
    def create_pathway_graphs(self, correlation_matrix: np.ndarray, 
                            gene_names: List[str], threshold: float = 0.3) -> Dict[str, nx.Graph]:
        """Create pathway-specific gene interaction graphs."""
        
        for pathway_name, gene_indices in self.pathway_info.items():
            # Create subgraph for this pathway
            pathway_genes = [gene_names[i] for i in gene_indices if i < len(gene_names)]
            pathway_corr = correlation_matrix[np.ix_(gene_indices, gene_indices)]
            
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(pathway_genes)
            
            # Add edges based on correlation threshold
            for i, gene1 in enumerate(pathway_genes):
                for j, gene2 in enumerate(pathway_genes):
                    if i < j and abs(pathway_corr[i, j]) > threshold:
                        G.add_edge(gene1, gene2, weight=abs(pathway_corr[i, j]))
            
            self.pathway_graphs[pathway_name] = G
            
        return self.pathway_graphs
    
    def calculate_bottleneck_scores(self, expression_data: np.ndarray, 
                                  gene_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate bottleneck scores for genes in each pathway."""
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(expression_data.T)
        
        # Create pathway graphs
        self.create_pathway_graphs(correlation_matrix, gene_names)
        
        pathway_bottlenecks = {}
        
        for pathway_name, graph in self.pathway_graphs.items():
            if len(graph.nodes()) == 0:
                continue
                
            bottleneck_scores = {}
            
            # Calculate various centrality measures
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            
            # Calculate flow betweenness (approximation of bottleneck importance)
            try:
                flow_betweenness = nx.edge_betweenness_centrality(graph)
                # Convert edge betweenness to node scores
                node_flow_scores = {}
                for node in graph.nodes():
                    connected_edges = graph.edges(node)
                    if connected_edges:
                        node_flow_scores[node] = np.mean([flow_betweenness.get(edge, 0) 
                                                        for edge in connected_edges])
                    else:
                        node_flow_scores[node] = 0
            except:
                node_flow_scores = {node: 0 for node in graph.nodes()}
            
            # Combine scores with weights
            for node in graph.nodes():
                combined_score = (
                    0.4 * betweenness.get(node, 0) +
                    0.3 * closeness.get(node, 0) +
                    0.2 * eigenvector.get(node, 0) +
                    0.1 * node_flow_scores.get(node, 0)
                )
                bottleneck_scores[node] = combined_score
            
            pathway_bottlenecks[pathway_name] = bottleneck_scores
        
        self.bottleneck_scores = pathway_bottlenecks
        return pathway_bottlenecks
    
    def identify_critical_bottlenecks(self, top_k: int = 10) -> Dict[str, List[str]]:
        """Identify top bottleneck genes for each pathway."""
        
        critical_bottlenecks = {}
        
        for pathway_name, scores in self.bottleneck_scores.items():
            # Sort genes by bottleneck score
            sorted_genes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            critical_bottlenecks[pathway_name] = [gene for gene, score in sorted_genes[:top_k]]
        
        return critical_bottlenecks
    
    def calculate_pathway_dysregulation(self, expression_data: np.ndarray, 
                                      gene_names: List[str], 
                                      sample_labels: np.ndarray) -> Dict[str, float]:
        """Calculate pathway-level dysregulation scores."""
        
        pathway_dysregulation = {}
        
        for pathway_name, gene_indices in self.pathway_info.items():
            # Get expression data for pathway genes
            valid_indices = [i for i in gene_indices if i < expression_data.shape[1]]
            if not valid_indices:
                continue
                
            pathway_expression = expression_data[:, valid_indices]
            
            # Calculate variance in expression across samples
            pathway_variance = np.mean(np.var(pathway_expression, axis=0))
            
            # Calculate correlation with sample labels (if available)
            label_correlation = 0
            if len(np.unique(sample_labels)) > 1:
                pathway_mean_expr = np.mean(pathway_expression, axis=1)
                try:
                    label_correlation = abs(pearsonr(pathway_mean_expr, sample_labels)[0])
                except:
                    label_correlation = 0
            
            # Combined dysregulation score
            dysregulation_score = 0.7 * pathway_variance + 0.3 * label_correlation
            pathway_dysregulation[pathway_name] = dysregulation_score
        
        return pathway_dysregulation
    
    def analyze_temporal_changes(self, temporal_data: Dict[int, np.ndarray], 
                               gene_names: List[str]) -> Dict[str, Dict[int, float]]:
        """Analyze temporal changes in pathway activity."""
        
        pathway_temporal_scores = {}
        
        for pathway_name, gene_indices in self.pathway_info.items():
            temporal_scores = {}
            valid_indices = [i for i in gene_indices if i < len(gene_names)]
            
            if not valid_indices:
                continue
            
            timepoint_expressions = {}
            for timepoint, data in temporal_data.items():
                pathway_expr = data[:, valid_indices]
                timepoint_expressions[timepoint] = np.mean(pathway_expr, axis=1)
            
            # Calculate rate of change between timepoints
            sorted_timepoints = sorted(timepoint_expressions.keys())
            for i in range(1, len(sorted_timepoints)):
                t1, t2 = sorted_timepoints[i-1], sorted_timepoints[i]
                expr1, expr2 = timepoint_expressions[t1], timepoint_expressions[t2]
                
                # Calculate correlation and change magnitude
                try:
                    correlation = pearsonr(expr1, expr2)[0]
                    change_magnitude = np.mean(np.abs(expr2 - expr1))
                    temporal_score = (1 - correlation) * change_magnitude
                except:
                    temporal_score = 0
                
                temporal_scores[t2] = temporal_score
            
            pathway_temporal_scores[pathway_name] = temporal_scores
        
        return pathway_temporal_scores
    
    def calculate_gene_pathway_importance(self, attention_weights: torch.Tensor, 
                                        gene_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate gene importance within each pathway using attention weights."""
        
        # Convert attention weights to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        pathway_gene_importance = {}
        
        for pathway_name, gene_indices in self.pathway_info.items():
            gene_importance = {}
            
            for idx in gene_indices:
                if idx < len(gene_names) and idx < attention_weights.shape[-1]:
                    gene_name = gene_names[idx]
                    # Average attention across all samples and heads
                    importance = np.mean(attention_weights[..., idx])
                    gene_importance[gene_name] = importance
            
            pathway_gene_importance[pathway_name] = gene_importance
        
        return pathway_gene_importance
    
    def visualize_pathway_network(self, pathway_name: str, 
                                save_path: Optional[str] = None) -> None:
        """Visualize pathway network with bottleneck genes highlighted."""
        
        if pathway_name not in self.pathway_graphs:
            print(f"Pathway {pathway_name} not found in graphs")
            return
        
        graph = self.pathway_graphs[pathway_name]
        bottleneck_scores = self.bottleneck_scores.get(pathway_name, {})
        
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Node colors based on bottleneck scores
        node_colors = [bottleneck_scores.get(node, 0) for node in graph.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                             cmap='Reds', node_size=500, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title(f'Pathway Network: {pathway_name}\n(Node color intensity = Bottleneck Score)')
        plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), 
                    label='Bottleneck Score', shrink=0.8)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_pathway_report(self, expression_data: np.ndarray, 
                              gene_names: List[str], 
                              sample_labels: np.ndarray,
                              attention_weights: Optional[torch.Tensor] = None) -> pd.DataFrame:
        """Generate comprehensive pathway analysis report."""
        
        # Calculate all metrics
        bottleneck_scores = self.calculate_bottleneck_scores(expression_data, gene_names)
        critical_bottlenecks = self.identify_critical_bottlenecks()
        dysregulation_scores = self.calculate_pathway_dysregulation(
            expression_data, gene_names, sample_labels
        )
        
        if attention_weights is not None:
            gene_importance = self.calculate_gene_pathway_importance(
                attention_weights, gene_names
            )
        
        # Create report dataframe
        report_data = []
        
        for pathway_name in self.pathway_info.keys():
            n_genes = len(self.pathway_info[pathway_name])
            n_bottlenecks = len(critical_bottlenecks.get(pathway_name, []))
            dysregulation = dysregulation_scores.get(pathway_name, 0)
            
            # Top bottleneck genes
            top_bottlenecks = critical_bottlenecks.get(pathway_name, [])[:5]
            
            # Average importance if attention weights available
            avg_importance = 0
            if attention_weights is not None and pathway_name in gene_importance:
                avg_importance = np.mean(list(gene_importance[pathway_name].values()))
            
            report_data.append({
                'Pathway': pathway_name,
                'Total_Genes': n_genes,
                'Bottleneck_Genes': n_bottlenecks,
                'Dysregulation_Score': dysregulation,
                'Avg_Gene_Importance': avg_importance,
                'Top_Bottlenecks': ', '.join(top_bottlenecks)
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Dysregulation_Score', ascending=False)
        
        return report_df
    
    def save_analysis_results(self, filepath: str) -> None:
        """Save analysis results to file."""
        results = {
            'pathway_info': self.pathway_info,
            'bottleneck_scores': self.bottleneck_scores,
            'importance_scores': self.importance_scores
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Analysis results saved to {filepath}")


class BottleneckRanker:
    """Rank genes based on their bottleneck potential across multiple criteria."""
    
    def __init__(self):
        self.ranking_criteria = {}
    
    def add_ranking_criterion(self, name: str, scores: Dict[str, float], weight: float = 1.0):
        """Add a ranking criterion with associated weights."""
        self.ranking_criteria[name] = {'scores': scores, 'weight': weight}
    
    def calculate_composite_ranking(self) -> Dict[str, float]:
        """Calculate composite ranking across all criteria."""
        
        # Get all genes across all criteria
        all_genes = set()
        for criterion in self.ranking_criteria.values():
            all_genes.update(criterion['scores'].keys())
        
        composite_scores = {}
        
        for gene in all_genes:
            weighted_score = 0
            total_weight = 0
            
            for criterion in self.ranking_criteria.values():
                if gene in criterion['scores']:
                    weighted_score += criterion['scores'][gene] * criterion['weight']
                    total_weight += criterion['weight']
            
            if total_weight > 0:
                composite_scores[gene] = weighted_score / total_weight
        
        return composite_scores
    
    def get_top_bottlenecks(self, top_k: int = 50) -> List[Tuple[str, float]]:
        """Get top-ranked bottleneck genes."""
        composite_scores = self.calculate_composite_ranking()
        sorted_genes = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_genes[:top_k]


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    pathway_info = {
        'Glycolysis': list(range(0, 50)),
        'TCA_Cycle': list(range(50, 100)),
        'Oxidative_Phosphorylation': list(range(100, 200))
    }
    
    # Generate example expression data
    n_samples, n_genes = 100, 200
    expression_data = np.random.lognormal(1, 1, (n_samples, n_genes))
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    sample_labels = np.random.binomial(1, 0.5, n_samples)
    
    # Create analyzer
    analyzer = MetabolicPathwayAnalyzer(pathway_info)
    
    # Generate report
    report = analyzer.generate_pathway_report(expression_data, gene_names, sample_labels)
    print(report) 