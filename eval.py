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


class Evaluator:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.device = device
        
        # Initialize model
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
        """Load model with weights"""
        try:
            # Try to load using the new checkpoint format first
            from utils.util import load_checkpoint
            model = nn.yolo_v11_n(self.args.num_cls).to(self.device)
            load_checkpoint('./weights/best.pt', model, device=self.device)
            model = model.float()
        except:
            # Fall back to old format
            model = torch.load('./weights/v11_n.pt', map_location=self.device, weights_only=False)['model'].float()
        model.eval()
        return model
    
    def evaluate_accuracy(self):
        """Evaluate accuracy metrics (mAP, Precision, Recall)"""
        print("Evaluating accuracy metrics...")
        
        # Setup validation dataset
        dataset = Dataset(self.args, self.params, False)
        loader = data.DataLoader(dataset, batch_size=16,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=Dataset.collate_fn)
        
        # Initialize metrics storage
        iou_v = torch.linspace(0.5, 0.95, 10)
        n_iou = iou_v.numel()
        metric = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
        
        # Run validation
        for batch in tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP')):
            image = (batch["img"].to(self.device).float()) / 255
            for k in ["idx", "cls", "box"]:
                batch[k] = batch[k].to(self.device)
            
            outputs = util.non_max_suppression(self.model(image))
            metric = util.update_metrics(outputs, batch, n_iou, iou_v, metric)
        
        # Compute final metrics
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in metric.items()}
        stats.pop("target_img", None)
        
        if len(stats) and stats["tp"].any():
            result = util.compute_ap(tp=stats['tp'],
                                     conf=stats['conf'],
                                     pred=stats['pred_cls'],
                                     target=stats['target_cls'],
                                     plot=True,
                                     save_dir='weights/',
                                     names=self.params['names'])
            
            self.metrics['Precision'] = result['precision']
            self.metrics['Recall'] = result['recall']
            self.metrics['mAP50'] = result['mAP50']
            self.metrics['mAP'] = result['mAP50-95']
        else:
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