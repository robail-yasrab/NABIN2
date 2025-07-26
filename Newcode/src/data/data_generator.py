import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, List
import random
from scipy.stats import multivariate_normal


class MultiOmicDataGenerator:
    """Generate synthetic multi-omic data for pathway analysis."""
    
    def __init__(self, n_samples: int = 1000, n_genes: int = 1000, n_pathways: int = 50, seed: int = 42):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define metabolic pathways (simplified)
        self.pathway_genes = self._create_pathway_structure()
        self.bottleneck_genes = self._define_bottleneck_genes()
        
    def _create_pathway_structure(self) -> Dict[str, List[int]]:
        """Create a mapping of pathways to gene indices."""
        pathways = {}
        pathway_names = [
            "Glycolysis", "TCA_Cycle", "Oxidative_Phosphorylation", "Fatty_Acid_Synthesis",
            "Amino_Acid_Metabolism", "Pentose_Phosphate", "Gluconeogenesis", "Lipid_Metabolism",
            "Purine_Metabolism", "Pyrimidine_Metabolism", "Steroid_Biosynthesis", "Cholesterol_Metabolism"
        ]
        
        genes_per_pathway = self.n_genes // len(pathway_names)
        
        for i, pathway in enumerate(pathway_names):
            start_idx = i * genes_per_pathway
            end_idx = min((i + 1) * genes_per_pathway, self.n_genes)
            pathways[pathway] = list(range(start_idx, end_idx))
            
        return pathways
    
    def _define_bottleneck_genes(self) -> List[int]:
        """Define which genes are bottlenecks in metabolic pathways."""
        bottleneck_genes = []
        for pathway, genes in self.pathway_genes.items():
            # Select 10-20% of genes in each pathway as potential bottlenecks
            n_bottlenecks = max(1, len(genes) // 8)
            bottlenecks = random.sample(genes, n_bottlenecks)
            bottleneck_genes.extend(bottlenecks)
        return bottleneck_genes
    
    def generate_cna_data(self) -> pd.DataFrame:
        """Generate Copy Number Alteration data."""
        # CNA values typically range from 0 (deletion) to 4+ (amplification), normal is 2
        cna_data = np.random.normal(2.0, 0.5, (self.n_samples, self.n_genes))
        
        # Add some systematic alterations for bottleneck genes
        for gene_idx in self.bottleneck_genes[:len(self.bottleneck_genes)//2]:
            # Some samples have deletions
            deletion_samples = np.random.choice(self.n_samples, size=self.n_samples//10, replace=False)
            cna_data[deletion_samples, gene_idx] = np.random.normal(0.5, 0.2, len(deletion_samples))
            
            # Some samples have amplifications
            amp_samples = np.random.choice(self.n_samples, size=self.n_samples//10, replace=False)
            cna_data[amp_samples, gene_idx] = np.random.normal(4.0, 0.5, len(amp_samples))
        
        cna_data = np.clip(cna_data, 0, 6)  # Biological constraints
        
        gene_names = [f"Gene_{i}" for i in range(self.n_genes)]
        sample_names = [f"Sample_{i}" for i in range(self.n_samples)]
        
        return pd.DataFrame(cna_data, columns=gene_names, index=sample_names)
    
    def generate_mutation_data(self) -> pd.DataFrame:
        """Generate mutation data (binary: 0=wild-type, 1=mutated)."""
        # Lower baseline mutation rate
        mutation_rate = 0.05
        mutation_data = np.random.binomial(1, mutation_rate, (self.n_samples, self.n_genes))
        
        # Higher mutation rates for bottleneck genes
        for gene_idx in self.bottleneck_genes:
            mutation_data[:, gene_idx] = np.random.binomial(1, 0.15, self.n_samples)
        
        gene_names = [f"Gene_{i}" for i in range(self.n_genes)]
        sample_names = [f"Sample_{i}" for i in range(self.n_samples)]
        
        return pd.DataFrame(mutation_data, columns=gene_names, index=sample_names)
    
    def generate_mrna_data(self, cna_data: pd.DataFrame, mutation_data: pd.DataFrame) -> pd.DataFrame:
        """Generate mRNA expression data correlated with CNA and mutations."""
        # Base expression levels
        mrna_data = np.random.lognormal(1.0, 1.0, (self.n_samples, self.n_genes))
        
        # Correlate with CNA (copy number affects expression)
        for i in range(self.n_samples):
            for j in range(self.n_genes):
                cna_effect = (cna_data.iloc[i, j] - 2.0) * 0.3  # Scaling factor
                mrna_data[i, j] *= (1 + cna_effect)
                
                # Mutations can affect expression
                if mutation_data.iloc[i, j] == 1:
                    # Mutations can increase or decrease expression
                    mut_effect = np.random.choice([-0.5, 0.5]) * np.random.uniform(0.2, 0.8)
                    mrna_data[i, j] *= (1 + mut_effect)
        
        # Add pathway-level correlations for bottleneck genes
        for pathway, genes in self.pathway_genes.items():
            pathway_bottlenecks = [g for g in genes if g in self.bottleneck_genes]
            if len(pathway_bottlenecks) > 1:
                # Create correlated expression for bottleneck genes
                correlation_strength = 0.3
                for i in range(len(pathway_bottlenecks) - 1):
                    gene1, gene2 = pathway_bottlenecks[i], pathway_bottlenecks[i + 1]
                    correlation = correlation_strength * mrna_data[:, gene1]
                    noise = np.random.normal(0, 0.1, self.n_samples)
                    mrna_data[:, gene2] = 0.7 * mrna_data[:, gene2] + 0.3 * correlation + noise
        
        mrna_data = np.maximum(mrna_data, 0.01)  # Ensure positive values
        
        gene_names = [f"Gene_{i}" for i in range(self.n_genes)]
        sample_names = [f"Sample_{i}" for i in range(self.n_samples)]
        
        return pd.DataFrame(mrna_data, columns=gene_names, index=sample_names)
    
    def generate_clinical_data(self, n_timepoints: int = 5) -> pd.DataFrame:
        """Generate clinical timepoint data."""
        clinical_features = [
            "age", "tumor_size", "grade", "stage", "survival_months",
            "treatment_response", "metastasis", "recurrence"
        ]
        
        clinical_data = []
        sample_names = []
        
        for sample_idx in range(self.n_samples):
            for timepoint in range(n_timepoints):
                sample_name = f"Sample_{sample_idx}_T{timepoint}"
                sample_names.append(sample_name)
                
                # Generate correlated clinical data over time
                if timepoint == 0:  # Baseline
                    age = np.random.normal(60, 15)
                    tumor_size = np.random.exponential(3)
                    grade = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                    stage = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
                    survival_months = 0
                    treatment_response = 0
                    metastasis = 0
                    recurrence = 0
                else:
                    # Temporal progression
                    age += 0.5  # Age increases
                    tumor_size *= np.random.normal(1.0, 0.1)  # Size can change
                    survival_months = timepoint * 6  # 6 months per timepoint
                    
                    # Treatment response (improves over time for some patients)
                    treatment_response = min(1.0, timepoint * 0.2 + np.random.normal(0, 0.1))
                    
                    # Metastasis and recurrence probabilities increase over time
                    metastasis_prob = min(0.8, timepoint * 0.1)
                    recurrence_prob = min(0.6, timepoint * 0.08)
                    
                    metastasis = np.random.binomial(1, metastasis_prob)
                    recurrence = np.random.binomial(1, recurrence_prob)
                
                clinical_row = [
                    age, tumor_size, grade, stage, survival_months,
                    treatment_response, metastasis, recurrence
                ]
                clinical_data.append(clinical_row)
        
        clinical_df = pd.DataFrame(clinical_data, columns=clinical_features, index=sample_names)
        return clinical_df
    
    def generate_outcome_labels(self, mrna_data: pd.DataFrame) -> pd.DataFrame:
        """Generate outcome labels based on bottleneck gene expression."""
        # Create binary outcome based on bottleneck gene expression patterns
        outcomes = []
        
        for i in range(self.n_samples):
            # Calculate bottleneck score based on expression of bottleneck genes
            bottleneck_expr = mrna_data.iloc[i, self.bottleneck_genes].values
            bottleneck_score = np.mean(np.log2(bottleneck_expr + 1))
            
            # Add some noise and create binary outcome
            noise = np.random.normal(0, 0.5)
            final_score = bottleneck_score + noise
            
            # Binary classification: good vs poor outcome
            outcome = 1 if final_score > np.median([bottleneck_score + np.random.normal(0, 0.5) 
                                                  for _ in range(100)]) else 0
            outcomes.append(outcome)
        
        sample_names = [f"Sample_{i}" for i in range(self.n_samples)]
        return pd.DataFrame({"outcome": outcomes}, index=sample_names)
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete multi-omic dataset."""
        print("Generating CNA data...")
        cna_data = self.generate_cna_data()
        
        print("Generating mutation data...")
        mutation_data = self.generate_mutation_data()
        
        print("Generating mRNA expression data...")
        mrna_data = self.generate_mrna_data(cna_data, mutation_data)
        
        print("Generating clinical data...")
        clinical_data = self.generate_clinical_data()
        
        print("Generating outcome labels...")
        outcome_data = self.generate_outcome_labels(mrna_data)
        
        return {
            "cna": cna_data,
            "mutations": mutation_data,
            "mrna": mrna_data,
            "clinical": clinical_data,
            "outcomes": outcome_data,
            "pathway_info": {
                "pathways": self.pathway_genes,
                "bottleneck_genes": self.bottleneck_genes
            }
        }


# Example usage
if __name__ == "__main__":
    generator = MultiOmicDataGenerator(n_samples=500, n_genes=1000)
    dataset = generator.generate_complete_dataset()
    
    print("Dataset shapes:")
    for key, data in dataset.items():
        if isinstance(data, pd.DataFrame):
            print(f"{key}: {data.shape}")
        else:
            print(f"{key}: {type(data)}") 