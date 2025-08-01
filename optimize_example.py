"""
Example script demonstrating how to use the optimization module directly
"""

import torch
from nets import nn
from utils import optimization
from utils.util import device

def optimize_yolo_model():
    """Example of optimizing a YOLO model"""
    print("Loading YOLOv11 model...")
    
    # Create model
    model = nn.yolo_v11_n(num_cls=80).to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 1. Dynamic Quantization Example
    print("\n1. Applying Dynamic Quantization...")
    quantizer = optimization.QuantizationOptimizer(model)
    quantized_model = quantizer.dynamic_quantization()
    print("Dynamic quantization applied successfully!")
    
    # Save quantized model
    quantizer.save_quantized_model("weights/dynamic_quantized_model.pt")
    print("Quantized model saved to weights/dynamic_quantized_model.pt")
    
    # Compare sizes
    size_info = optimization.compare_model_sizes(model, quantized_model)
    print(f"Model size compression ratio: {size_info['compression_ratio']:.2f}x")
    print(f"Memory savings: {size_info['memory_savings_mb']:.2f} MB")
    
    # 2. Magnitude Pruning Example
    print("\n2. Applying Magnitude Pruning...")
    pruner = optimization.PruningOptimizer(model)
    pruned_model = pruner.magnitude_pruning(sparsity=0.3)
    print("Magnitude pruning applied successfully!")
    
    # Save pruned model
    torch.save(pruned_model.state_dict(), "weights/pruned_model.pt")
    print("Pruned model saved to weights/pruned_model.pt")
    
    # Compare sizes
    size_info = optimization.compare_model_sizes(model, pruned_model)
    print(f"Model size compression ratio: {size_info['compression_ratio']:.2f}x")
    print(f"Memory savings: {size_info['memory_savings_mb']:.2f} MB")
    
    # 3. Benchmark comparison
    print("\n3. Benchmarking inference speed...")
    
    # Benchmark original model
    original_benchmark = optimization.benchmark_inference_speed(
        model, input_shape=(1, 3, 640, 640), num_iterations=50, device=str(device)
    )
    
    # Benchmark quantized model
    quantized_benchmark = optimization.benchmark_inference_speed(
        quantized_model, input_shape=(1, 3, 640, 640), num_iterations=50, device=str(device)
    )
    
    print("Original model:")
    print(f"  Average inference time: {original_benchmark['avg_time_ms']:.2f} ms")
    print(f"  FPS: {original_benchmark['fps']:.2f}")
    
    print("Quantized model:")
    print(f"  Average inference time: {quantized_benchmark['avg_time_ms']:.2f} ms")
    print(f"  FPS: {quantized_benchmark['fps']:.2f}")
    
    speedup = quantized_benchmark['fps'] / original_benchmark['fps']
    print(f"Speedup: {speedup:.2f}x")
    
    print("\nOptimization examples completed!")

if __name__ == "__main__":
    optimize_yolo_model()