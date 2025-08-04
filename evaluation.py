import os
import time
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import yaml

from nets import nn
from utils import util
from utils.util import device
from utils.dataset import Dataset
from torch.utils import data

# Import main module functions to reuse existing interfaces
import main

class Evaluator:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.device = device
        
        # Initialize model using existing smart_load_model interface
        self.model = self.load_model()
        
        # Initialize metrics
        self.metrics = {
            'mAP': 0.0,
            'mAP50': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'FPS': 0.0,
            'InferenceTime': 0.0
        }
        
    def load_model(self):
        """Load model using existing smart_load_model interface"""
        try:
            # Use the existing smart_load_model function from main.py
            weights_path = getattr(self.args, 'weights', './weights/best.pt')
            if not os.path.exists(weights_path):
                weights_path = './weights/combined_weights.pt'
            if not os.path.exists(weights_path):
                weights_path = './weights/combined_model.pt'
                
            model = main.smart_load_model(weights_path, self.args.num_cls, self.device)
            print(f"Model loaded successfully using smart_load_model from {weights_path}")
            return model
        except Exception as e:
            print(f"Error loading model with smart_load_model: {e}")
            # Fallback to basic model creation
            model = nn.yolo_v11_n(self.args.num_cls).to(self.device)
            print("Created new model as fallback")
            return model.float()
    
    def evaluate_accuracy(self):
        """Evaluate accuracy metrics using existing validate function"""
        print("Evaluating accuracy metrics using existing validate function...")
        
        # Create a copy of args with plot enabled for evaluation
        eval_args = argparse.Namespace()
        for attr in dir(self.args):
            if not attr.startswith('_'):
                setattr(eval_args, attr, getattr(self.args, attr))
        eval_args.plot = True
        
        # Use the existing validate function from main.py
        try:
            m_pre, m_rec, map50, mean_ap = main.validate(eval_args, self.params, self.model)
            
            self.metrics['Precision'] = m_pre
            self.metrics['Recall'] = m_rec
            self.metrics['mAP50'] = map50
            self.metrics['mAP'] = mean_ap
            
            print(f"Validation completed: Precision={m_pre:.4f}, Recall={m_rec:.4f}, mAP50={map50:.4f}, mAP={mean_ap:.4f}")
            
        except Exception as e:
            print(f"Error during validation: {e}")
            self.metrics['Precision'] = 0.0
            self.metrics['Recall'] = 0.0
            self.metrics['mAP50'] = 0.0
            self.metrics['mAP'] = 0.0
            
        return self.metrics
    
    def evaluate_speed(self, num_images=100, image_size=640):
        """Evaluate speed metrics (FPS, inference time)"""
        print("Evaluating speed metrics...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
        
        # Set model to eval mode safely
        try:
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not set model to eval mode: {e}")
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Measure inference time
        times = []
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        for i in tqdm(range(num_images), desc="Measuring inference speed"):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        self.metrics['InferenceTime'] = avg_time * 1000  # Convert to milliseconds
        self.metrics['FPS'] = fps
        
        return self.metrics
    
    def run_evaluation(self, mode='full'):
        """Run evaluation based on mode"""
        print(f"Running evaluation in {mode} mode...")
        
        if mode in ['full', 'accuracy']:
            self.evaluate_accuracy()
        
        if mode in ['full', 'speed']:
            self.evaluate_speed()
        
        return self.metrics
    
    def print_report(self):
        """Print detailed evaluation report"""
        print("\n" + "="*60)
        print("                    EVALUATION REPORT")
        print("="*60)
        print(f"{'Metric':<20} {'Value':<15} {'Unit':<15}")
        print("-"*60)
        print(f"{'mAP@0.5:0.95':<20} {self.metrics['mAP']:<15.4f} {'-'}")
        print(f"{'mAP@0.5':<20} {self.metrics['mAP50']:<15.4f} {'-'}")
        print(f"{'Precision':<20} {self.metrics['Precision']:<15.4f} {'-'}")
        print(f"{'Recall':<20} {self.metrics['Recall']:<15.4f} {'-'}")
        print(f"{'Inference Time':<20} {self.metrics['InferenceTime']:<15.2f} {'ms'}")
        print(f"{'FPS':<20} {self.metrics['FPS']:<15.2f} {'images/sec'}")
        print("="*60)
        
        # Save metrics to file
        with open('evaluation_report.txt', 'w') as f:
            f.write("Evaluation Report\n")
            f.write("================\n")
            f.write(f"mAP@0.5:0.95: {self.metrics['mAP']:.4f}\n")
            f.write(f"mAP@0.5: {self.metrics['mAP50']:.4f}\n")
            f.write(f"Precision: {self.metrics['Precision']:.4f}\n")
            f.write(f"Recall: {self.metrics['Recall']:.4f}\n")
            f.write(f"Inference Time: {self.metrics['InferenceTime']:.2f} ms\n")
            f.write(f"FPS: {self.metrics['FPS']:.2f} images/sec\n")
        
        print("\nReport saved to evaluation_report.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cls', type=int, default=80)
    parser.add_argument('--inp-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-dir', type=str, default='COCO')
    parser.add_argument('--weights', type=str, help='Path to model weights')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'accuracy', 'speed', 'quick'],
                        help='Evaluation mode: full, accuracy, speed, or quick')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images for speed evaluation')
    
    args = parser.parse_args()
    
    # Load parameters
    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(args, params)
    
    # Quick mode - reduced evaluation for faster results
    if args.mode == 'quick':
        args.num_images = 10  # Reduced number of images for speed test
    
    # Run evaluation
    metrics = evaluator.run_evaluation(mode=args.mode if args.mode != 'quick' else 'full')
    
    # Print report
    evaluator.print_report()

if __name__ == "__main__":
    main()