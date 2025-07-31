#!/usr/bin/env python3
"""
Test script for the evaluation module
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import torch
        print("✓ PyTorch import successful")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import eval
        print("✓ Evaluation module import successful")
    except ImportError as e:
        print(f"✗ Evaluation module import failed: {e}")
        return False
    
    try:
        from nets import nn
        print("✓ Neural network module import successful")
    except ImportError as e:
        print(f"✗ Neural network module import failed: {e}")
        return False
    
    try:
        from utils import util
        print("✓ Utility module import successful")
    except ImportError as e:
        print(f"✗ Utility module import failed: {e}")
        return False
    
    return True

def test_evaluator_creation():
    """Test that evaluator can be created"""
    try:
        import eval
        import yaml
        
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.num_cls = 80
                self.inp_size = 640
                self.batch_size = 16
                self.data_dir = 'COCO'
        
        # Load parameters
        with open('utils/args.yaml', errors='ignore') as f:
            params = yaml.safe_load(f)
        
        # Create evaluator
        args = MockArgs()
        evaluator = eval.Evaluator(args, params)
        print("✓ Evaluator creation successful")
        return True
    except Exception as e:
        print(f"✗ Evaluator creation failed: {e}")
        return False

def main():
    print("Running evaluation module tests...\n")
    
    tests = [
        test_imports,
        test_evaluator_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())