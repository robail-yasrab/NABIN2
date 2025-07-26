import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import networkx as nx
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns

class MultiOmicsDataGenerator:
    """Generate synthetic multi-omics data for metabolic pathway bottleneck analysis."""
    
    def __init__(self, n_samples: int = 1000, n_genes: int = 500, n_pathways: int = 20, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            n_samples: Number of patient samples
            n_genes: Number of genes to simulate
            n_pathways: Number of metabolic pathways
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define pathway structure and bottleneck genes
        self.pathway_genes = self._create_pathway_structure()
        self.bottleneck_genes = self._identify_bottleneck_genes()
        
    def _create_pathway_structure(self) -> Dict[str, List[int]]:
        """Create synthetic metabolic pathway structure."""
        pathway_genes = {}
        genes_per_pathway = self.n_genes // self.n_pathways
        
        for i in range(self.n_pathways):
            start_idx = i * genes_per_pathway
            end_idx = min((i + 1) * genes_per_pathway, self.n_genes)
            pathway_genes[f'pathway_{i}'] = list(range(start_idx, end_idx))
            
        return pathway_genes
    
    def _identify_bottleneck_genes(self) -> List[int]:
        """Identify genes that act as bottlenecks in pathways."""
        bottleneck_genes = []
        for pathway, genes in self.pathway_genes.items():
            # Select 1-2 genes per pathway as bottlenecks
            n_bottlenecks = np.random.randint(1, 3)
            bottlenecks = np.random.choice(genes, n_bottlenecks, replace=False)
            bottleneck_genes.extend(bottlenecks)
        return sorted(list(set(bottleneck_genes)))
    
    def generate_mrna_expression(self) -> np.ndarray:
        """Generate mRNA expression data with pathway structure."""
        # Base expression levels
        expression = np.random.lognormal(mean=2, sigma=1, size=(self.n_samples, self.n_genes))
        
        # Add pathway co-expression patterns
        for pathway, genes in self.pathway_genes.items():
            # Generate correlated expression within pathways
            pathway_factor = np.random.normal(0, 1, self.n_samples)
            for gene_idx in genes:
                expression[:, gene_idx] *= (1 + 0.3 * pathway_factor)
        
        # Enhance bottleneck gene expression patterns
        for gene_idx in self.bottleneck_genes:
            # Bottleneck genes have more variable expression
            expression[:, gene_idx] *= np.random.lognormal(0, 0.5, self.n_samples)
        
        return np.log2(expression + 1)  # Log-transform
    
    def generate_cna_data(self) -> np.ndarray:
        """Generate Copy Number Alteration data."""
        # CNA values: -2 (homozygous deletion), -1 (heterozygous deletion), 
        # 0 (normal), 1 (gain), 2 (amplification)
        cna_probs = [0.05, 0.15, 0.6, 0.15, 0.05]  # Probabilities for each state
        cna_values = [-2, -1, 0, 1, 2]
        
        cna_data = np.random.choice(cna_values, size=(self.n_samples, self.n_genes), p=cna_probs)
        
        # Bottleneck genes have higher probability of alterations
        for gene_idx in self.bottleneck_genes:
            # Increase probability of amplification/deletion for bottleneck genes
            altered_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
            cna_data[:, gene_idx] = np.random.choice(cna_values, size=self.n_samples, p=altered_probs)
        
        return cna_data.astype(np.float32)
    
    def generate_mutation_data(self) -> np.ndarray:
        """Generate binary mutation data."""
        # Base mutation rate: 5%
        mutation_prob = 0.05
        mutation_data = np.random.binomial(1, mutation_prob, size=(self.n_samples, self.n_genes))
        
        # Bottleneck genes have higher mutation rates
        for gene_idx in self.bottleneck_genes:
            mutation_data[:, gene_idx] = np.random.binomial(1, 0.15, size=self.n_samples)
        
        return mutation_data.astype(np.float32)
    
    def generate_clinical_timepoints(self) -> Dict[str, np.ndarray]:
        """Generate clinical timepoint data (survival, progression, etc.)."""
        # Overall survival (months)
        base_survival = np.random.exponential(24, self.n_samples)  # 24 months median
        
        # Progression-free survival
        base_pfs = np.random.exponential(12, self.n_samples)  # 12 months median
        
        # Response scores (0-100)
        response_scores = np.random.beta(2, 2, self.n_samples) * 100
        
        return {
            'overall_survival': base_survival,
            'progression_free_survival': base_pfs,
            'response_score': response_scores,
            'survival_event': (np.random.random(self.n_samples) < 0.3).astype(np.float32)  # 30% event rate
        }
    
    def create_pathway_graph(self) -> nx.Graph:
        """Create a graph representation of metabolic pathways."""
        G = nx.Graph()
        
        # Add nodes (genes)
        for i in range(self.n_genes):
            G.add_node(i, is_bottleneck=(i in self.bottleneck_genes))
        
        # Add edges within pathways
        for pathway, genes in self.pathway_genes.items():
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i+1:]:
                    # Add edges with some probability
                    if np.random.random() < 0.3:
                        G.add_edge(gene1, gene2, pathway=pathway)
        
        # Add inter-pathway connections for bottleneck genes
        bottleneck_connections = []
        for i, gene1 in enumerate(self.bottleneck_genes):
            for gene2 in self.bottleneck_genes[i+1:]:
                if np.random.random() < 0.1:  # 10% chance of connection
                    G.add_edge(gene1, gene2, type='bottleneck_connection')
                    bottleneck_connections.append((gene1, gene2))
        
        return G
    
    def generate_complete_dataset(self) -> Dict[str, torch.Tensor]:
        """Generate complete multi-omics dataset."""
        print("Generating multi-omics dataset...")
        
        # Generate each data modality
        mrna_data = self.generate_mrna_expression()
        cna_data = self.generate_cna_data()
        mutation_data = self.generate_mutation_data()
        clinical_data = self.generate_clinical_timepoints()
        
        # Create pathway graph
        pathway_graph = self.create_pathway_graph()
        
        # Convert to tensors
        dataset = {
            'mrna_expression': torch.FloatTensor(mrna_data),
            'cna': torch.FloatTensor(cna_data),
            'mutations': torch.FloatTensor(mutation_data),
            'overall_survival': torch.FloatTensor(clinical_data['overall_survival']),
            'progression_free_survival': torch.FloatTensor(clinical_data['progression_free_survival']),
            'response_score': torch.FloatTensor(clinical_data['response_score']),
            'survival_event': torch.FloatTensor(clinical_data['survival_event']),
            'pathway_graph': pathway_graph,
            'bottleneck_genes': self.bottleneck_genes,
            'pathway_genes': self.pathway_genes
        }
        
        # Add gene names
        gene_names = [f'GENE_{i:04d}' for i in range(self.n_genes)]
        dataset['gene_names'] = gene_names
        
        print(f"Dataset generated successfully!")
        print(f"- Samples: {self.n_samples}")
        print(f"- Genes: {self.n_genes}")
        print(f"- Pathways: {self.n_pathways}")
        print(f"- Bottleneck genes: {len(self.bottleneck_genes)}")
        
        return dataset
    
    def visualize_data_summary(self, dataset: Dict[str, torch.Tensor], save_path: str = None):
        """Create visualization of the generated dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # mRNA expression distribution
        axes[0, 0].hist(dataset['mrna_expression'].numpy().flatten(), bins=50, alpha=0.7)
        axes[0, 0].set_title('mRNA Expression Distribution')
        axes[0, 0].set_xlabel('Log2(Expression + 1)')
        axes[0, 0].set_ylabel('Frequency')
        
        # CNA distribution
        cna_counts = np.bincount(dataset['cna'].numpy().flatten().astype(int) + 2)
        axes[0, 1].bar(range(-2, 3), cna_counts)
        axes[0, 1].set_title('CNA Distribution')
        axes[0, 1].set_xlabel('CNA Value')
        axes[0, 1].set_ylabel('Count')
        
        # Mutation frequency
        mutation_freq = dataset['mutations'].sum(dim=0)
        axes[0, 2].hist(mutation_freq.numpy(), bins=30, alpha=0.7)
        axes[0, 2].set_title('Mutation Frequency per Gene')
        axes[0, 2].set_xlabel('Number of Mutations')
        axes[0, 2].set_ylabel('Number of Genes')
        
        # Survival data
        axes[1, 0].hist(dataset['overall_survival'].numpy(), bins=30, alpha=0.7, label='Overall Survival')
        axes[1, 0].hist(dataset['progression_free_survival'].numpy(), bins=30, alpha=0.7, label='PFS')
        axes[1, 0].set_title('Survival Distributions')
        axes[1, 0].set_xlabel('Time (months)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Response scores
        axes[1, 1].hist(dataset['response_score'].numpy(), bins=30, alpha=0.7)
        axes[1, 1].set_title('Response Score Distribution')
        axes[1, 1].set_xlabel('Response Score')
        axes[1, 1].set_ylabel('Frequency')
        
        # Pathway graph statistics
        G = dataset['pathway_graph']
        degree_sequence = [d for n, d in G.degree()]
        axes[1, 2].hist(degree_sequence, bins=20, alpha=0.7)
        axes[1, 2].set_title('Pathway Graph: Node Degree Distribution')
        axes[1, 2].set_xlabel('Node Degree')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Generate sample dataset
    generator = MultiOmicsDataGenerator(n_samples=1000, n_genes=500, n_pathways=20)
    dataset = generator.generate_complete_dataset()
    
    # Visualize the data
    generator.visualize_data_summary(dataset, 'data_summary.png')
    
    # Save dataset
    torch.save(dataset, 'multiomics_dataset.pt')
    print("Dataset saved as 'multiomics_dataset.pt'") 