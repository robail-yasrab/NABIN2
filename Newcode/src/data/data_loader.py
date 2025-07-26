import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MultiOmicDataset(Dataset):
    """PyTorch Dataset for multi-omic data."""
    
    def __init__(self, cna_data: torch.Tensor, mutation_data: torch.Tensor, 
                 mrna_data: torch.Tensor, clinical_data: torch.Tensor, 
                 outcomes: torch.Tensor, sample_ids: List[str]):
        self.cna_data = cna_data
        self.mutation_data = mutation_data
        self.mrna_data = mrna_data
        self.clinical_data = clinical_data
        self.outcomes = outcomes
        self.sample_ids = sample_ids
        
    def __len__(self):
        return len(self.cna_data)
    
    def __getitem__(self, idx):
        return {
            'cna': self.cna_data[idx],
            'mutations': self.mutation_data[idx],
            'mrna': self.mrna_data[idx],
            'clinical': self.clinical_data[idx],
            'outcome': self.outcomes[idx],
            'sample_id': self.sample_ids[idx]
        }


class MultiOmicDataLoader:
    """Data loader for multi-omic datasets."""
    
    def __init__(self, batch_size: int = 32, test_size: float = 0.2, 
                 val_size: float = 0.1, random_state: int = 42):
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Scalers for normalization
        self.cna_scaler = StandardScaler()
        self.mrna_scaler = StandardScaler()
        self.clinical_scaler = StandardScaler()
        
    def preprocess_data(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, torch.Tensor]:
        """Preprocess multi-omic data for model training."""
        
        # Extract data
        cna_df = dataset['cna']
        mutation_df = dataset['mutations']
        mrna_df = dataset['mrna']
        clinical_df = dataset['clinical']
        outcome_df = dataset['outcomes']
        
        print("Preprocessing data...")
        
        # Align samples (handle clinical timepoints)
        base_samples = [f"Sample_{i}" for i in range(len(cna_df))]
        
        # For clinical data, use baseline timepoint (T0)
        clinical_baseline = clinical_df[clinical_df.index.str.endswith('_T0')].copy()
        clinical_baseline.index = clinical_baseline.index.str.replace('_T0', '')
        
        # Ensure all dataframes have the same samples
        common_samples = list(set(cna_df.index) & set(mutation_df.index) & 
                            set(mrna_df.index) & set(clinical_baseline.index) & 
                            set(outcome_df.index))
        
        print(f"Found {len(common_samples)} common samples")
        
        # Reindex all dataframes to common samples
        cna_aligned = cna_df.loc[common_samples]
        mutation_aligned = mutation_df.loc[common_samples]
        mrna_aligned = mrna_df.loc[common_samples]
        clinical_aligned = clinical_baseline.loc[common_samples]
        outcome_aligned = outcome_df.loc[common_samples]
        
        # Handle missing values
        cna_aligned = cna_aligned.fillna(cna_aligned.median())
        mutation_aligned = mutation_aligned.fillna(0)
        mrna_aligned = mrna_aligned.fillna(mrna_aligned.median())
        clinical_aligned = clinical_aligned.fillna(clinical_aligned.median())
        
        # Normalize data
        cna_normalized = self.cna_scaler.fit_transform(cna_aligned.values)
        mrna_normalized = self.mrna_scaler.fit_transform(np.log2(mrna_aligned.values + 1))
        clinical_normalized = self.clinical_scaler.fit_transform(clinical_aligned.values)
        
        # Convert to tensors
        cna_tensor = torch.FloatTensor(cna_normalized)
        mutation_tensor = torch.FloatTensor(mutation_aligned.values)
        mrna_tensor = torch.FloatTensor(mrna_normalized)
        clinical_tensor = torch.FloatTensor(clinical_normalized)
        outcome_tensor = torch.LongTensor(outcome_aligned['outcome'].values)
        
        print(f"Data shapes after preprocessing:")
        print(f"CNA: {cna_tensor.shape}")
        print(f"Mutations: {mutation_tensor.shape}")
        print(f"mRNA: {mrna_tensor.shape}")
        print(f"Clinical: {clinical_tensor.shape}")
        print(f"Outcomes: {outcome_tensor.shape}")
        
        return {
            'cna': cna_tensor,
            'mutations': mutation_tensor,
            'mrna': mrna_tensor,
            'clinical': clinical_tensor,
            'outcomes': outcome_tensor,
            'sample_ids': common_samples
        }
    
    def create_temporal_features(self, clinical_df: pd.DataFrame) -> torch.Tensor:
        """Create temporal features from clinical timepoint data."""
        # Group by sample and create temporal sequences
        temporal_features = []
        
        sample_groups = {}
        for idx in clinical_df.index:
            if '_T' in idx:
                sample_id = idx.split('_T')[0]
                timepoint = int(idx.split('_T')[1])
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = {}
                sample_groups[sample_id][timepoint] = clinical_df.loc[idx].values
        
        # Create sequences for each sample
        max_timepoints = 5
        for sample_id in sorted(sample_groups.keys()):
            timepoints = sample_groups[sample_id]
            sequence = []
            
            for t in range(max_timepoints):
                if t in timepoints:
                    sequence.append(timepoints[t])
                else:
                    # Pad with last available timepoint or zeros
                    if sequence:
                        sequence.append(sequence[-1])
                    else:
                        sequence.append(np.zeros(len(clinical_df.columns)))
            
            temporal_features.append(sequence)
        
        # Convert to tensor [n_samples, n_timepoints, n_features]
        temporal_tensor = torch.FloatTensor(temporal_features)
        temporal_normalized = self.clinical_scaler.fit_transform(
            temporal_tensor.reshape(-1, temporal_tensor.shape[-1])
        )
        temporal_tensor = torch.FloatTensor(
            temporal_normalized.reshape(temporal_tensor.shape)
        )
        
        return temporal_tensor
    
    def prepare_dataloaders(self, dataset: Dict[str, pd.DataFrame], 
                          use_temporal: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test dataloaders."""
        
        # Preprocess data
        processed_data = self.preprocess_data(dataset)
        
        # Handle temporal clinical data if requested
        if use_temporal:
            clinical_tensor = self.create_temporal_features(dataset['clinical'])
        else:
            clinical_tensor = processed_data['clinical']
        
        # Split data
        n_samples = len(processed_data['sample_ids'])
        indices = np.arange(n_samples)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state,
            stratify=processed_data['outcomes']
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=self.val_size/(1-self.test_size), 
            random_state=self.random_state,
            stratify=processed_data['outcomes'][train_val_idx]
        )
        
        print(f"Data splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Create datasets
        def create_subset(indices):
            return MultiOmicDataset(
                cna_data=processed_data['cna'][indices],
                mutation_data=processed_data['mutations'][indices],
                mrna_data=processed_data['mrna'][indices],
                clinical_data=clinical_tensor[indices],
                outcomes=processed_data['outcomes'][indices],
                sample_ids=[processed_data['sample_ids'][i] for i in indices]
            )
        
        train_dataset = create_subset(train_idx)
        val_dataset = create_subset(val_idx)
        test_dataset = create_subset(test_idx)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dimensions(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Get feature dimensions for each data modality."""
        return {
            'cna_dim': dataset['cna'].shape[1],
            'mutation_dim': dataset['mutations'].shape[1],
            'mrna_dim': dataset['mrna'].shape[1],
            'clinical_dim': dataset['clinical'].shape[1],
            'n_classes': len(dataset['outcomes']['outcome'].unique())
        }
    
    def save_preprocessed_data(self, dataset: Dict[str, pd.DataFrame], 
                             filepath: str) -> None:
        """Save preprocessed data to file."""
        processed_data = self.preprocess_data(dataset)
        torch.save({
            'data': processed_data,
            'scalers': {
                'cna_scaler': self.cna_scaler,
                'mrna_scaler': self.mrna_scaler,
                'clinical_scaler': self.clinical_scaler
            }
        }, filepath)
        print(f"Preprocessed data saved to {filepath}")
    
    def load_preprocessed_data(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load preprocessed data from file."""
        checkpoint = torch.load(filepath)
        self.cna_scaler = checkpoint['scalers']['cna_scaler']
        self.mrna_scaler = checkpoint['scalers']['mrna_scaler']
        self.clinical_scaler = checkpoint['scalers']['clinical_scaler']
        print(f"Preprocessed data loaded from {filepath}")
        return checkpoint['data']


# Example usage
if __name__ == "__main__":
    from data_generator import MultiOmicDataGenerator
    
    # Generate synthetic data
    generator = MultiOmicDataGenerator(n_samples=500, n_genes=1000)
    dataset = generator.generate_complete_dataset()
    
    # Create data loaders
    data_loader = MultiOmicDataLoader(batch_size=32)
    train_loader, val_loader, test_loader = data_loader.prepare_dataloaders(dataset)
    
    print("Data loaders created successfully!")
    
    # Test a batch
    batch = next(iter(train_loader))
    print("\nBatch example:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {len(value)} samples") 