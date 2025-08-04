"""
Model optimization module for YOLOv11
Supports various optimization techniques including quantization, pruning, and distillation
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Any, Optional, Union
import warnings

try:
    import torch.quantization as quant
    TORCH_QUANT_AVAILABLE = True
except ImportError:
    TORCH_QUANT_AVAILABLE = False
    warnings.warn("PyTorch quantization module not available. Quantization features will be disabled.")

class QuantizationOptimizer:
    """Quantization optimization class supporting various quantization methods"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize quantization optimizer
        
        Args:
            model: PyTorch model to be quantized
        """
        self.model = model
        self.quantized_model = None
        self.is_quantized = False
        
    def dynamic_quantization(self, 
                           dtype: torch.dtype = torch.qint8,
                           quant_modules: Optional[list] = None) -> nn.Module:
        """
        Apply dynamic quantization to the model
        
        Args:
            dtype: Data type for quantization (default: torch.qint8)
            quant_modules: List of module types to quantize (default: [nn.Linear, nn.LSTM])
            
        Returns:
            Quantized model
        """
        if not TORCH_QUANT_AVAILABLE:
            raise RuntimeError("PyTorch quantization is not available")
            
        # Check if quantization engines are available
        available_backends = torch.backends.quantized.supported_engines
        if 'none' in available_backends and len(available_backends) == 1:
            print("Warning: No quantization engines available. Dynamic quantization may not work.")
            
        # Default modules to quantize
        if quant_modules is None:
            quant_modules = [nn.Linear, nn.LSTM]
            
        # Create a copy of the model for quantization
        self.quantized_model = copy.deepcopy(self.model)
        
        # Apply dynamic quantization
        try:
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.quantized_model, 
                {type(m) for m in self.quantized_model.modules() if type(m) in quant_modules},
                dtype=dtype
            )
        except RuntimeError as e:
            if "NoQEngine" in str(e):
                print("Dynamic quantization failed due to missing quantization engine.")
                print("This may be due to your PyTorch installation not including quantization support.")
                print("Consider reinstalling PyTorch with full quantization support.")
                raise e
            else:
                raise e
        
        self.is_quantized = True
        return self.quantized_model
        
    def static_quantization_prepare(self, 
                                  backend: str = 'qnnpack',
                                  quant_modules: Optional[list] = None) -> nn.Module:
        """
        Prepare model for static quantization (first step)
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            quant_modules: List of module types to prepare for quantization
            
        Returns:
            Prepared model
        """
        if not TORCH_QUANT_AVAILABLE:
            raise RuntimeError("PyTorch quantization is not available")
            
        # Check available backends and set appropriate one
        available_backends = torch.backends.quantized.supported_engines
        if backend not in available_backends:
            print(f"Requested backend '{backend}' not available. Available backends: {available_backends}")
            # Use the first available backend that's not 'none'
            backend = next((b for b in available_backends if b != 'none'), 'qnnpack')
            print(f"Using backend: {backend}")
            
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Default modules to quantize
        if quant_modules is None:
            quant_modules = [nn.Conv2d, nn.Linear]
            
        # Create a copy of the model for quantization
        self.quantized_model = copy.deepcopy(self.model)
        
        # Set quantization configuration
        self.quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare model for static quantization
        torch.quantization.prepare(self.quantized_model, inplace=True)
        
        return self.quantized_model
        
    def static_quantization_convert(self, 
                                  calibration_data_loader=None,
                                  num_calibration_batches=10) -> nn.Module:
        """
        Convert prepared model to statically quantized model (second step)
        
        Args:
            calibration_data_loader: DataLoader for calibration (optional)
            num_calibration_batches: Number of batches to use for calibration
            
        Returns:
            Quantized model
        """
        if not TORCH_QUANT_AVAILABLE:
            raise RuntimeError("PyTorch quantization is not available")
            
        if self.quantized_model is None:
            raise RuntimeError("Model not prepared for static quantization. Call static_quantization_prepare first.")
            
        # if no self.quantized_model.quant()
        if "quant" not in self.quantized_model._modules:
            self.quantized_model.quant = torch.quantization.QuantStub()
            self.quantized_model.dequant = torch.quantization.DeQuantStub()
            

        # If calibration data is provided, run calibration
        if calibration_data_loader is not None:
            self.quantized_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(calibration_data_loader):
                    if i >= num_calibration_batches:
                        break
                    # Handle different batch formats
                    if isinstance(batch, dict) and "img" in batch:
                        images = batch["img"]
                    elif isinstance(batch, (list, tuple)):
                        images = batch[0]  # Assume first element is images
                    else:
                        images = batch
                        
                    # Move to device and normalize if needed
                    if hasattr(images, 'to'):
                        images = images.to(next(self.quantized_model.parameters()).device)
                    if images.dtype == torch.uint8:
                        images = images.float() / 255.0
                        
                    _ = self.quantized_model(images)
        
        # Convert to quantized model
        try:
            torch.quantization.convert(self.quantized_model, inplace=True)
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized::" in str(e):
                print("Static quantization conversion failed due to missing quantization engine.")
                print("This may be due to your PyTorch installation not including quantization support.")
                print("Consider reinstalling PyTorch with full quantization support.")
                raise e
            else:
                raise e
                
        self.is_quantized = True
        
        return self.quantized_model
        
    def quantization_aware_training_prepare(self, 
                                          backend: str = 'fbgemm',
                                          quant_modules: Optional[list] = None) -> nn.Module:
        """
        Prepare model for quantization aware training
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            quant_modules: List of module types to prepare for QAT
            
        Returns:
            Model prepared for QAT
        """
        if not TORCH_QUANT_AVAILABLE:
            raise RuntimeError("PyTorch quantization is not available")
            
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Default modules to quantize
        if quant_modules is None:
            quant_modules = [nn.Conv2d, nn.Linear]
            
        # Create a copy of the model for QAT
        self.quantized_model = copy.deepcopy(self.model)
        
        # Set QAT configuration
        self.quantized_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        
        # Prepare model for QAT
        torch.quantization.prepare_qat(self.quantized_model, inplace=True)
        
        return self.quantized_model
        
    def save_quantized_model(self, filepath: str):
        """
        Save quantized model to file
        
        Args:
            filepath: Path to save the quantized model
        """
        if not self.is_quantized or self.quantized_model is None:
            raise RuntimeError("No quantized model available. Apply quantization first.")
            
        # For quantized models, we need to save the entire model, not just state dict
        # This is because quantized models have special modules that need to be preserved
        torch.save(self.quantized_model, filepath)
        
    def load_quantized_model(self, filepath: str) -> nn.Module:
        """
        Load quantized model from file
        
        Args:
            filepath: Path to the quantized model file
            
        Returns:
            Loaded quantized model
        """
        if not TORCH_QUANT_AVAILABLE:
            raise RuntimeError("PyTorch quantization is not available")
            
        # For quantized models saved with torch.save(model), we can load directly
        self.quantized_model = torch.load(filepath, map_location='cpu')
        self.is_quantized = True
        return self.quantized_model


class PruningOptimizer:
    """Model pruning optimization class"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize pruning optimizer
        
        Args:
            model: PyTorch model to be pruned
        """
        self.model = model
        self.pruned_model = None
        self.is_pruned = False
        
    def magnitude_pruning(self, 
                         sparsity: float = 0.5,
                         prune_modules: Optional[list] = None) -> nn.Module:
        """
        Apply magnitude-based pruning to the model
        
        Args:
            sparsity: Sparsity level (0.0 to 1.0)
            prune_modules: List of module types to prune (default: [nn.Linear, nn.Conv2d])
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise RuntimeError("PyTorch pruning utilities not available")
            
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0.0 and 1.0")
            
        # Default modules to prune
        if prune_modules is None:
            prune_modules = [nn.Linear, nn.Conv2d]
            
        # Create a copy of the model for pruning
        self.pruned_model = copy.deepcopy(self.model)
        
        # Apply magnitude pruning
        for module in self.pruned_model.modules():
            if type(module) in prune_modules:
                if hasattr(module, 'weight'):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=sparsity)
        
        self.is_pruned = True
        return self.pruned_model
        
    def structured_pruning(self, 
                         sparsity: float = 0.5,
                         prune_modules: Optional[list] = None) -> nn.Module:
        """
        Apply structured pruning (channel/filter pruning) to the model
        
        Args:
            sparsity: Sparsity level (0.0 to 1.0)
            prune_modules: List of module types to prune (default: [nn.Conv2d])
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise RuntimeError("PyTorch pruning utilities not available")
            
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError("Sparsity must be between 0.0 and 1.0")
            
        # Default modules to prune (structured pruning typically applied to Conv layers)
        if prune_modules is None:
            prune_modules = [nn.Conv2d]
            
        # Create a copy of the model for pruning
        self.pruned_model = copy.deepcopy(self.model)
        
        # Apply structured pruning
        for module in self.pruned_model.modules():
            if type(module) in prune_modules:
                if hasattr(module, 'weight'):
                    # Prune entire channels/filters
                    prune.random_structured(module, name='weight', amount=sparsity, dim=0)
        
        self.is_pruned = True
        return self.pruned_model
        
    def remove_pruning_hooks(self):
        """
        Remove pruning hooks and make pruning permanent
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise RuntimeError("PyTorch pruning utilities not available")
            
        if self.pruned_model is None:
            raise RuntimeError("No pruned model available")
            
        # Remove pruning hooks permanently
        for module in self.pruned_model.modules():
            if hasattr(module, 'weight_orig'):  # Check if module is pruned
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_orig'):
                prune.remove(module, 'bias')


class DistillationOptimizer:
    """Model distillation optimization class"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        """
        Initialize distillation optimizer
        
        Args:
            teacher_model: Pre-trained teacher model (larger, more accurate)
            student_model: Student model to be trained (smaller, more efficient)
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_model.eval()  # Teacher model should be in eval mode
        
    def compute_distillation_loss(self, 
                                inputs: torch.Tensor,
                                temperature: float = 3.0,
                                alpha: float = 0.7) -> torch.Tensor:
        """
        Compute distillation loss combining hard and soft targets
        
        Args:
            inputs: Input tensor
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for hard loss)
            
        Returns:
            Combined distillation loss
        """
        # Get teacher predictions (with softmax and temperature)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            if isinstance(teacher_outputs, (list, tuple)):
                teacher_outputs = teacher_outputs[0]  # Take first output if multiple
            soft_teacher = torch.softmax(teacher_outputs / temperature, dim=1)
        
        # Get student predictions
        student_outputs = self.student_model(inputs)
        if isinstance(student_outputs, (list, tuple)):
            student_outputs = student_outputs[0]  # Take first output if multiple
            
        # Hard loss (standard cross-entropy with true labels)
        # Note: This would need true labels which are not passed here
        # In practice, you would pass labels and compute hard_loss = F.cross_entropy(student_outputs, labels)
        hard_loss = 0  # Placeholder - in practice you would compute this with true labels
        
        # Soft loss (KL divergence between softened distributions)
        soft_student = torch.softmax(student_outputs / temperature, dim=1)
        soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(
            torch.log(soft_student), soft_teacher) * (temperature ** 2)
        
        # Combined loss
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return total_loss


class ModelOptimizer:
    """Unified model optimization interface"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize model optimizer
        
        Args:
            model: PyTorch model to optimize
        """
        self.model = model
        self.quantizer = QuantizationOptimizer(model) if TORCH_QUANT_AVAILABLE else None
        self.pruner = PruningOptimizer(model)
        
    def apply_optimization(self, 
                          opt_type: str,
                          **kwargs) -> nn.Module:
        """
        Apply specified optimization technique
        
        Args:
            opt_type: Type of optimization ('quantization', 'pruning', 'distillation')
            **kwargs: Additional arguments for the optimization method
            
        Returns:
            Optimized model
        """
        if opt_type == 'quantization':
            if not TORCH_QUANT_AVAILABLE:
                raise RuntimeError("Quantization not available")
            if self.quantizer is None:
                self.quantizer = QuantizationOptimizer(self.model)
            return self.quantizer.dynamic_quantization(**kwargs)
            
        elif opt_type == 'pruning':
            return self.pruner.magnitude_pruning(**kwargs)
            
        elif opt_type == 'distillation':
            raise ValueError("Distillation requires a separate teacher model. Use DistillationOptimizer directly.")
            
        else:
            raise ValueError(f"Unknown optimization type: {opt_type}")
            
    def save_optimized_model(self, filepath: str, opt_type: str):
        """
        Save optimized model to file
        
        Args:
            filepath: Path to save the optimized model
            opt_type: Type of optimization applied
        """
        if opt_type == 'quantization':
            if self.quantizer is None:
                raise RuntimeError("No quantized model available")
            self.quantizer.save_quantized_model(filepath)
        elif opt_type == 'pruning':
            if self.pruner.pruned_model is None:
                raise RuntimeError("No pruned model available")
            torch.save(self.pruner.pruned_model.state_dict(), filepath)
        else:
            raise ValueError(f"Unknown optimization type: {opt_type}")


# Utility functions
def compare_model_sizes(original_model: nn.Module, optimized_model: nn.Module) -> Dict[str, float]:
    """
    Compare sizes of original and optimized models
    
    Args:
        original_model: Original PyTorch model
        optimized_model: Optimized PyTorch model
        
    Returns:
        Dictionary with size comparison metrics
    """
    # Calculate model sizes
    original_size = sum(p.numel() for p in original_model.parameters())
    optimized_size = sum(p.numel() for p in optimized_model.parameters())
    
    # For quantized models, also estimate memory footprint
    original_memory = sum(p.numel() * p.element_size() for p in original_model.parameters())
    
    # Estimate quantized memory (assuming 8-bit quantization)
    quantized_memory = sum(p.numel() for p in optimized_model.parameters())  # 1 byte per parameter
    
    return {
        'original_params': original_size,
        'optimized_params': optimized_size,
        'compression_ratio': original_size / optimized_size if optimized_size > 0 else float('inf'),
        'original_memory_mb': original_memory / (1024 * 1024),
        'estimated_quantized_memory_mb': quantized_memory / (1024 * 1024),
        'memory_savings_mb': (original_memory - quantized_memory) / (1024 * 1024)
    }


def benchmark_inference_speed(model: nn.Module, 
                            input_shape: tuple = (1, 3, 640, 640),
                            num_iterations: int = 100,
                            device: str = 'cpu') -> Dict[str, float]:
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
        device: Device to run benchmark on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Move model to device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device_obj)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    return {
        'total_time': total_time,
        'avg_time_ms': avg_time * 1000,
        'fps': fps
    }