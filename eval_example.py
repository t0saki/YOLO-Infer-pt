#!/usr/bin/env python3
"""
Example usage of the evaluation module
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("YOLOv11 Evaluation Examples")
    print("==========================")
    
    print("\n1. Full evaluation (accuracy and speed):")
    print("   python main.py --evaluate")
    
    print("\n2. Accuracy-only evaluation:")
    print("   python main.py --evaluate --eval-mode accuracy")
    
    print("\n3. Speed-only evaluation:")
    print("   python main.py --evaluate --eval-mode speed")
    
    print("\n4. Quick evaluation (reduced dataset for faster results):")
    print("   python main.py --evaluate --eval-mode quick")
    
    print("\n5. Direct evaluation script with more options:")
    print("   python eval.py --mode full")
    print("   python eval.py --mode accuracy")
    print("   python eval.py --mode speed --num-images 50")
    
    print("\n6. Using the evaluation module programmatically:")
    print("""
    import eval
    import yaml
    
    # Create arguments
    class Args:
        num_cls = 80
        inp_size = 640
        batch_size = 16
        data_dir = 'COCO'
    
    # Load parameters
    with open('utils/args.yaml') as f:
        params = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = eval.Evaluator(Args(), params)
    
    # Run full evaluation
    metrics = evaluator.run_evaluation(mode='full')
    
    # Print report
    evaluator.print_report()
    """)

if __name__ == "__main__":
    main()