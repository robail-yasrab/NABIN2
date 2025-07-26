#!/usr/bin/env python3
"""
Quick Installation Test for Multi-Omic Pathway Analysis

This script tests that all components are working correctly.
Run this after installing requirements to verify everything is set up properly.
"""

import sys
import os
import traceback


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'scipy', 'tqdm', 'networkx'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("All required packages imported successfully!")
    return True


def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from src.data.data_generator import MultiOmicDataGenerator
        print("✓ Data generator")
        
        from src.data.data_loader import MultiOmicDataLoader
        print("✓ Data loader")
        
        from src.models.multimodal_transformer import MultiOmicTransformer
        print("✓ Multi-modal transformer")
        
        from src.training.trainer import MultiOmicTrainer
        print("✓ Trainer")
        
        from src.analysis.pathway_analyzer import MetabolicPathwayAnalyzer
        print("✓ Pathway analyzer")
        
        from src.utils.visualization import plot_multi_omic_overview
        print("✓ Visualization utilities")
        
        print("All project modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        traceback.print_exc()
        return False


def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    
    try:
        from src.data.data_generator import MultiOmicDataGenerator
        
        # Small test dataset
        generator = MultiOmicDataGenerator(n_samples=50, n_genes=100, seed=42)
        dataset = generator.generate_complete_dataset()
        
        # Check dataset structure
        expected_keys = ['cna', 'mutations', 'mrna', 'clinical', 'outcomes', 'pathway_info']
        for key in expected_keys:
            if key not in dataset:
                print(f"✗ Missing key: {key}")
                return False
        
        print("✓ Dataset generated with correct structure")
        print(f"✓ CNA shape: {dataset['cna'].shape}")
        print(f"✓ Mutations shape: {dataset['mutations'].shape}")
        print(f"✓ mRNA shape: {dataset['mrna'].shape}")
        print(f"✓ Clinical shape: {dataset['clinical'].shape}")
        print(f"✓ Outcomes shape: {dataset['outcomes'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and basic forward pass."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.models.multimodal_transformer import create_multimodal_transformer
        
        # Model configuration
        dataset_info = {
            'cna_dim': 100,
            'mutation_dim': 100,
            'mrna_dim': 100,
            'clinical_dim': 8,
            'n_classes': 2
        }
        
        pathway_info = {
            'test_pathway_1': list(range(0, 50)),
            'test_pathway_2': list(range(50, 100))
        }
        
        config = {
            'hidden_dim': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1
        }
        
        # Create model
        model = create_multimodal_transformer(dataset_info, pathway_info, config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        batch = {
            'cna': torch.randn(4, 100),
            'mutations': torch.randn(4, 100),
            'mrna': torch.randn(4, 100),
            'clinical': torch.randn(4, 8)
        }
        
        with torch.no_grad():
            outputs = model(batch)
        
        print("✓ Forward pass successful")
        print(f"✓ Classification logits shape: {outputs['classification_logits'].shape}")
        print(f"✓ Bottleneck scores shape: {outputs['bottleneck_scores'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation/testing failed: {e}")
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")
    
    try:
        from src.data.data_generator import MultiOmicDataGenerator
        from src.data.data_loader import MultiOmicDataLoader
        
        # Generate small dataset
        generator = MultiOmicDataGenerator(n_samples=100, n_genes=50, seed=42)
        dataset = generator.generate_complete_dataset()
        
        # Create data loader
        data_loader = MultiOmicDataLoader(batch_size=8)
        train_loader, val_loader, test_loader = data_loader.prepare_dataloaders(dataset)
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        # Test batch loading
        batch = next(iter(train_loader))
        print("✓ Batch loading successful")
        
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader testing failed: {e}")
        traceback.print_exc()
        return False


def test_pathway_analysis():
    """Test pathway analysis functionality."""
    print("\nTesting pathway analysis...")
    
    try:
        import numpy as np
        from src.analysis.pathway_analyzer import MetabolicPathwayAnalyzer
        
        # Create test data
        pathway_info = {
            'test_pathway_1': list(range(0, 25)),
            'test_pathway_2': list(range(25, 50))
        }
        
        analyzer = MetabolicPathwayAnalyzer(pathway_info)
        
        # Generate test expression data
        expression_data = np.random.lognormal(1, 1, (50, 50))
        gene_names = [f"Gene_{i}" for i in range(50)]
        sample_labels = np.random.binomial(1, 0.5, 50)
        
        # Test bottleneck calculation
        bottleneck_scores = analyzer.calculate_bottleneck_scores(expression_data, gene_names)
        print(f"✓ Bottleneck scores calculated for {len(bottleneck_scores)} pathways")
        
        # Test pathway report
        report = analyzer.generate_pathway_report(expression_data, gene_names, sample_labels)
        print(f"✓ Pathway report generated with {len(report)} pathways")
        
        return True
        
    except Exception as e:
        print(f"✗ Pathway analysis testing failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all installation tests."""
    print("=" * 60)
    print("MULTI-OMIC PATHWAY ANALYSIS - INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Data Generation", test_data_generation),
        ("Model Creation", test_model_creation),
        ("Data Loader", test_data_loader),
        ("Pathway Analysis", test_pathway_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"TEST: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Installation is successful.")
        print("You can now run the full analysis with:")
        print("   python examples/run_analysis.py")
        return True
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 