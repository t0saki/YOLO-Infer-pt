"""
YOLOv11 Next-Gen Object Detection using PyTorch
Main entry point for training, validation, inference, and evaluation
"""

import os
import csv
import cv2
import tqdm
import yaml
import torch
import argparse
import warnings
import numpy as np
from torch.utils import data
from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_ as clip

from nets import nn
from utils import util
from utils.dataset import Dataset

from utils.util import device, backend

# Conditional import for evaluation module
# try:
import evaluation
EVAL_MODULE_AVAILABLE = True
# except ImportError:
#     EVAL_MODULE_AVAILABLE = False
#     print("Warning: evaluation module not found. Evaluation functionality will be limited.")

# Conditional import for optimization module
try:
    from utils import optimization
    OPTIMIZATION_MODULE_AVAILABLE = True
except ImportError:
    OPTIMIZATION_MODULE_AVAILABLE = False
    print("Warning: optimization module not found. Optimization functionality will be limited.")

def smart_load_model(weights_path, num_classes, target_device=None):
    """
    Intelligently load a model from various checkpoint formats
    Supports both state dicts and complete model objects
    
    Args:
        weights_path: Path to the model file
        num_classes: Number of classes for the model
        target_device: Device to load the model to
        
    Returns:
        model: Loaded model ready for use
    """
    if target_device is None:
        target_device = device
    
    print(f"Smart loading model from {weights_path}")
    
    # First try to load the checkpoint
    try:
        # Try with weights_only=True first
        checkpoint = torch.load(weights_path, map_location=target_device, weights_only=True)
    except:
        try:
            # If that fails, try with weights_only=False
            checkpoint = torch.load(weights_path, map_location=target_device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
    
    # Handle different checkpoint formats
    if hasattr(checkpoint, 'state_dict') or hasattr(checkpoint, 'detect') or hasattr(checkpoint, '__call__'):
        # This is a complete model object
        print("Detected complete model object")
        model = checkpoint.to(target_device) if hasattr(checkpoint, 'to') else checkpoint
        return model.float()
        
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_data = checkpoint['model']
            
            # Check if model_data is a complete model or state dict
            if hasattr(model_data, 'state_dict') or hasattr(model_data, 'detect') or hasattr(model_data, '__call__'):
                # Complete model object in checkpoint
                print("Detected complete model object in checkpoint dict")
                model = model_data.to(target_device) if hasattr(model_data, 'to') else model_data
                return model.float()
            elif isinstance(model_data, dict):
                # State dict in checkpoint
                print("Detected state dict in checkpoint dict")
                model = nn.yolo_v11_n(num_classes).to(target_device)
                model.load_state_dict(model_data)
                return model.float()
            else:
                print(f"Unknown model data format in checkpoint: {type(model_data)}")
                raise ValueError(f"Cannot handle model data of type: {type(model_data)}")
        else:
            # This might be a bare state dict
            print("Attempting to load as bare state dict")
            model = nn.yolo_v11_n(num_classes).to(target_device)
            model.load_state_dict(checkpoint)
            return model.float()
    else:
        print(f"Unknown checkpoint format: {type(checkpoint)}")
        raise ValueError(f"Cannot handle checkpoint of type: {type(checkpoint)}")

def train(args, params):
    """
    Train the YOLO model with all integrated features:
    - Checkpoint resume functionality
    - Error tolerance for batch processing
    - Advanced logging and metrics tracking
    - EMA (Exponential Moving Average) support
    - Distributed training support
    - Model optimization support
    """
    util.init_seeds()
    
    # Initialize model
    model = nn.yolo_v11_n(args.num_cls)
    
    # Initialize scaler for mixed precision training
    use_scaler = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device=device, enabled=use_scaler)
    
    # Load pretrained weights if specified (but not checkpoint resume)
    if hasattr(args, 'weights') and args.weights and not (hasattr(args, 'resume') and args.resume):
        from utils.util import load_ultralytics_weight
        model = load_ultralytics_weight(model, args.weights)
    model.to(device)
    
    # Setup distributed training if enabled
    if args.distributed:
        util.setup_ddp(args)
    
    # Freeze specific layers (e.g., DFL layer)
    util.freeze_layer(model)
    
    # Setup EMA for model averaging
    ema = util.EMA(model) if args.rank == 0 else None
    
    # Prepare dataset and dataloader
    sampler = None
    dataset = Dataset(args, params, True)
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    
    batch_size = args.batch_size // max(args.world_size, 1)
    loader = data.DataLoader(dataset, batch_size, sampler is None,
                             sampler, num_workers=8, pin_memory=True,
                             collate_fn=Dataset.collate_fn)
    
    # Setup optimizer and learning rate scheduler
    accumulate = max(round(64 / args.batch_size * args.world_size), 1)
    decay = params['decay'] * args.batch_size * accumulate / 64
    optimizer = util.smart_optimizer(args, model, decay)
    linear = lambda _: (max(1 / args.epochs, 0) * (1.0 - 0.01) + 0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear)
    
    # Initialize training state variables
    start_epoch = 0
    best_map = 0.0
    
    # If resuming, load the complete checkpoint state
    if hasattr(args, 'resume') and args.resume:
        try:
            from utils.util import load_checkpoint
            _, start_epoch, best_map = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, ema, device=device)
            print(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            # Handle checkpoint loading errors
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0
            best_map = 0.0
    
    # If not resuming or scheduler state wasn't loaded, reset scheduler
    if not (hasattr(args, 'resume') and args.resume):
        scheduler.last_epoch = -1
    
    # Setup loss function
    criterion = util.DetectionLoss(model)
    
    # Training loop variables
    opt_step = -1
    num_batch = len(loader)
    warm_up = max(round(3 * num_batch), 100)
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = params.get('checkpoint_dir', 'weights')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize error tolerance variables
    consecutive_errors = 0
    total_errors = 0
    
    # Setup logging
    log_file = os.path.join(checkpoint_dir, 'step.csv')
    with open(log_file, 'w') as log:
        if args.rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()
        
        # Training epochs loop
        for epoch in range(start_epoch, args.epochs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()
            
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            
            p_bar = enumerate(loader)
            # Disable mosaic augmentation for last 10 epochs
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False
            
            if args.rank == 0:
                print("\n" + "%11s" * 5 % ("Epoch", "GPU", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(enumerate(loader), total=num_batch)
            
            t_loss = None
            # Batch training loop
            for i, batch in p_bar:
                # Error tolerance mechanism
                enable_error_tolerance = params.get('enable_error_tolerance', True)
                max_consecutive_errors = params.get('max_consecutive_errors', 10)
                
                try:
                    glob_step = i + num_batch * epoch
                    # Warmup phase
                    if glob_step <= warm_up:
                        xi = [0, warm_up]
                        accumulate = max(1, int(np.interp(glob_step, xi, [1, 64 / args.batch_size]).round()))
                        for j, param_group in enumerate(optimizer.param_groups):
                            param_group["lr"] = np.interp(glob_step, xi, [0.0 if j == 0 else 0.0,
                                                                param_group["initial_lr"] * linear(epoch)])
                            
                            if "momentum" in param_group:
                                param_group["momentum"] = np.interp(glob_step, xi, [0.8, 0.937])
                    
                    # Prepare input data
                    images = batch["img"].to(device).float() / 255
                    
                    # Forward pass with device-specific optimizations
                    if device.type == 'cuda':
                        with torch.amp.autocast(device):
                            pred = model(images)
                            loss, loss_items = criterion(pred, batch)
                            # Check for NaN or inf losses
                            if torch.isnan(loss) or torch.isinf(loss):
                                raise ValueError(f"Loss is NaN or infinite: {loss}")
                            if args.distributed:
                                loss *= args.world_size
                            
                            t_loss = ((t_loss * i + loss_items) / (
                                        i + 1) if t_loss is not None else loss_items)
                    elif device.type == 'mps':
                        # MPS autocast support
                        with torch.amp.autocast('cpu'):  # Use CPU autocast for MPS
                            pred = model(images)
                            loss, loss_items = criterion(pred, batch)
                            # Check for NaN or inf losses
                            if torch.isnan(loss) or torch.isinf(loss):
                                raise ValueError(f"Loss is NaN or infinite: {loss}")
                            if args.distributed:
                                loss *= args.world_size
                            
                            t_loss = ((t_loss * i + loss_items) / (
                                        i + 1) if t_loss is not None else loss_items)
                    else:
                        pred = model(images)
                        loss, loss_items = criterion(pred, batch)
                        # Check for NaN or inf losses
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"Loss is NaN or infinite: {loss}")
                        if args.distributed:
                            loss *= args.world_size
                        
                        t_loss = ((t_loss * i + loss_items) / (
                                    i + 1) if t_loss is not None else loss_items)
                    
                    # Check for gradient explosion
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(f"Gradient explosion detected: loss={loss}")
                    
                    # Backward pass
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    # Check for gradient NaN or inf
                    has_nan_grads = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grads = True
                                break
                    
                    if has_nan_grads:
                        raise ValueError("Gradient contains NaN or infinite values")
                        
                    # Optimization step
                    if glob_step - opt_step >= accumulate:
                        if use_scaler:
                            scaler.unscale_(optimizer)
                        clip(model.parameters(), max_norm=10.0)
                        if use_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        if ema:
                            ema.update(model)
                        opt_step = glob_step
                        
                    # Reset consecutive errors counter on successful batch
                    consecutive_errors = 0
                    
                except Exception as e:
                    # Handle batch error with tolerance mechanism
                    if enable_error_tolerance:
                        consecutive_errors += 1
                        total_errors += 1
                        
                        # Write error info to log file
                        error_log_file = params.get('error_log_file', os.path.join(checkpoint_dir, 'error_batches.log'))
                        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
                        with open(error_log_file, 'a') as f:
                            f.write(f"Epoch: {epoch}, Batch: {i}, Error: {str(e)}\n")
                        
                        print(f"Warning: Skipping batch {i} in epoch {epoch} due to error: {str(e)}")
                        
                        # Zero gradients to prevent accumulation from failed batch
                        optimizer.zero_grad()
                        
                        # Check if we've exceeded max consecutive errors
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Error: Too many consecutive errors ({consecutive_errors}). Stopping training.")
                            raise RuntimeError(f"Exceeded maximum consecutive errors ({max_consecutive_errors})")
                    else:
                        # Re-raise the exception if error tolerance is disabled
                        raise e
                
                # Update progress bar
                if args.rank == 0:
                    fmt = "%11s" * 2 + "%11.4g" * 3
                    if device.type == 'cuda':
                        memory = f'{torch.cuda.memory_allocated() / 1e9:.3g}G'
                    elif device.type == 'mps':
                        memory = f'{torch.mps.current_allocated_memory() / 1e9:.3g}G'
                    else:
                        memory = 'CPU'
                    p_bar.set_description(fmt % (f"{epoch + 1}/{args.epochs}", memory, *t_loss))
            
            # Validation and checkpointing at end of epoch
            if args.rank == 0:
                m_pre, m_rec, map50, mean_map = validate(args, params, ema.ema if ema else model)
                box, cls, dfl = map(float, t_loss)
                
                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{box:.3f}'),
                                 'cls': str(f'{cls:.3f}'),
                                 'dfl': str(f'{dfl:.3f}'),
                                 'mAP': str(f'{mean_map:.3f}'),
                                 'mAP@50': str(f'{map50:.3f}'),
                                 'Recall': str(f'{m_rec:.3f}'),
                                 'Precision': str(f'{m_pre:.3f}')})
                log.flush()
                
                # Save checkpoints based on configuration
                if args.rank == 0:
                    from utils.util import save_checkpoint
                    
                    # Save last checkpoint if enabled
                    if params.get('save_last', True):
                        save_checkpoint(
                            epoch=epoch,
                            model=ema.ema if ema else model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            ema=ema,
                            best_map=best_map,
                            filename=os.path.join(checkpoint_dir, 'last.pt')
                        )
                    
                    # Save best checkpoint if enabled and current model is better
                    if params.get('save_best', True) and mean_map > best_map:
                        best_map = mean_map
                        save_checkpoint(
                            epoch=epoch,
                            model=ema.ema if ema else model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            ema=ema,
                            best_map=best_map,
                            filename=os.path.join(checkpoint_dir, 'best.pt')
                        )
                    
                    # Save periodic checkpoints based on frequency
                    checkpoint_freq = params.get('checkpoint_freq', 1)
                    if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                        save_checkpoint(
                            epoch=epoch,
                            model=ema.ema if ema else model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            ema=ema,
                            best_map=best_map,
                            filename=os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
                        )
            
            # Synchronize distributed processes
            if args.distributed:
                dist.barrier()
        
        # Cleanup distributed training
        if args.distributed:
            dist.destroy_process_group()
        
        print(f"Training complete. Total errors encountered: {total_errors}")

@torch.no_grad()
def validate(args, params, model=None):
    """
    Validate the model performance on validation dataset
    
    Args:
        args: Command line arguments
        params: Configuration parameters
        model: Model to validate (optional, will load from checkpoint if not provided)
        
    Returns:
        tuple: (precision, recall, mAP@0.5, mAP@0.5:0.95)
    """
    # Set quantization engine before loading model
    if device.type == 'cpu':
        torch.backends.quantized.engine = 'qnnpack'
    elif device.type == 'mps':
        torch.backends.quantized.engine = 'qnnpack'
    else:
        # For other devices, try to set a supported engine
        available_engines = torch.backends.quantized.supported_engines
        if 'qnnpack' in available_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif available_engines and available_engines[0] != 'none':
            torch.backends.quantized.engine = available_engines[0]

    # Verify that we have a valid quantization engine
    if torch.backends.quantized.engine == 'none':
        print("Warning: No quantization engine available. Quantized models may not work properly.")
    iou_v = torch.linspace(0.5, 0.95, 10)
    n_iou = iou_v.numel()
    
    metric = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    
    # Load model if not provided
    if not model:
        args.plot = True
        weights_path = args.weights if args.weights else os.path.join('weights', 'best.pt')
        try:
            model = smart_load_model(weights_path, args.num_cls, device)
            if hasattr(model, 'fuse'):
                model = model.fuse()
        except Exception as e:
            print(f"Error loading model using smart_load_model: {e}")
            # Fallback to checkpoint loading
            try:
                from utils.util import load_checkpoint
                temp_model = nn.yolo_v11_n(args.num_cls).to(device)
                load_checkpoint(weights_path, temp_model, device=device)
                model = temp_model.float()
                if hasattr(model, 'fuse'):
                    model = model.fuse()
            except Exception as e2:
                print(f"Error with fallback loading: {e2}")
                raise e2
    
    # For quantized models, calling eval() might cause issues
    try:
        model.eval()
    except Exception as e:
        print(f"Warning: Could not set model to eval mode: {e}")
        # Continue anyway since we're doing validation
    
    dataset = Dataset(args, params, False)
    loader = data.DataLoader(dataset, batch_size=16,
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    
    # Validation loop
    for batch in tqdm.tqdm(loader, desc=('%10s' * 5) % (
    '', 'precision', 'recall', 'mAP50', 'mAP')):
        image = batch["img"].to(device)
        # For quantized models, we need to use float, not half
        # Check if this is a quantized model by looking for quantized modules
        is_quantized_model = any(
            hasattr(module, '_packed_params') or 
            'Quantized' in type(module).__name__ or
            hasattr(module, 'qconfig') and module.qconfig is not None
            for module in model.modules()
        )
        
        if is_quantized_model:
            # Quantized models work with float32 inputs
            image = image.float() / 255
        else:
            # For regular models, use half precision on CUDA, float on CPU/MPS
            if device.type == 'cuda':
                image = image.half() / 255
            else:
                image = image.float() / 255
        for k in ["idx", "cls", "box"]:
            batch[k] = batch[k].to(device)
        
        outputs = util.non_max_suppression(model(image))

        metric = util.update_metrics(outputs, batch, n_iou, iou_v, metric, device)

    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in metric.items()}
    stats.pop("target_img", None)
    if len(stats) and stats["tp"].any():
        result = util.compute_ap(tp=stats['tp'],
                                 conf=stats['conf'],
                                 pred=stats['pred_cls'],
                                 target=stats['target_cls'],
                                 plot=args.plot,
                                 save_dir='weights/',
                                 names=params['names'])
        
        m_pre = result['precision']
        m_rec = result['recall']
        map50 = result['mAP50']
        mean_ap = result['mAP50-95']
    else:
        m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0
    
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    
    model.float()
    return m_pre, m_rec, map50, mean_ap

@torch.no_grad()
def inference(args, params):
    """
    Run inference on video input using the trained model
    
    Args:
        args: Command line arguments
        params: Configuration parameters
    """
    # Set quantization engine before loading model
    if args.device == 'cpu' or (args.device is None and device.type == 'cpu'):
        torch.backends.quantized.engine = 'qnnpack'
    elif torch.backends.mps.is_available():
        torch.backends.quantized.engine = 'qnnpack'
    else:
        # For other devices, try to set a supported engine
        available_engines = torch.backends.quantized.supported_engines
        if 'qnnpack' in available_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif available_engines and available_engines[0] != 'none':
            torch.backends.quantized.engine = available_engines[0]
    
    # Verify that we have a valid quantization engine
    if torch.backends.quantized.engine == 'none':
        print("Warning: No quantization engine available. Quantized models may not work properly.")
    
    weights_path = args.weights if args.weights else os.path.join('.', 'weights', 'best.pt')
    
    # Use the smart model loading function
    try:
        model = smart_load_model(weights_path, args.num_cls, device)
        print(f"Model loaded successfully using smart_load_model")
    except Exception as e:
        print(f"Error loading model with smart_load_model: {e}")
        return
    
    # Check if this is a quantized model
    is_quantized_model = any(
        hasattr(module, '_packed_params') or 
        'Quantized' in type(module).__name__ or
        hasattr(module, 'qconfig') and module.qconfig is not None
        for module in model.modules()
    )
    
    # For regular (non-quantized) models, disable Conv layer fusion
    if not is_quantized_model:
        def disable_conv_fusion(module):
            """Recursively disable fusion for all Conv modules"""
            for name, child in module.named_children():
                if hasattr(child, 'unfuse') and callable(child.unfuse):
                    child.unfuse()
                    print(f"Disabled fusion for Conv module: {name}")
                elif hasattr(child, '_use_fused'):
                    child._use_fused = False
                    print(f"Disabled fusion flag for module: {name}")
                # Recursively process child modules
                disable_conv_fusion(child)
        
        print("Disabling Conv layer fusion to avoid forward_fuse errors...")
        disable_conv_fusion(model)
    else:
        print(f"Quantized model verification: {is_quantized_model}")
    
    # Set model to evaluation mode safely
    try:
        model.eval()
        print("Model set to evaluation mode successfully")
    except Exception as e:
        print(f"Warning: Could not set model to eval mode using .eval(): {e}")
        # For quantized models, manually set training mode
        if hasattr(model, 'training'):
            model.training = False
            print("Manually set model training mode to False")
        
        # For quantized models, recursively set evaluation mode
        def set_quantized_eval(module):
            """Safely set quantized model to eval mode"""
            if hasattr(module, 'training'):
                module.training = False
            if hasattr(module, 'children'):
                for child in module.children():
                    set_quantized_eval(child)
        
        if is_quantized_model:
            set_quantized_eval(model)
            print("Set quantized model to evaluation mode manually")
    
    print(f"Model loaded successfully. Model type: {type(model)}, Quantized: {is_quantized_model}")
    
    # Setup video capture
    camera = cv2.VideoCapture('input.mp4')
    
    # Get video properties
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer with fallback codecs
    try:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        # Test if the writer was created successfully
        if not out.isOpened():
            raise Exception("H264 codec failed")
    except:
        # Fallback to MP4V codec
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("mp4v codec failed")
        except:
            # Final fallback to XVID codec with .avi extension
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
            print("Using XVID codec, output file will be: output.avi")
    
    if not camera.isOpened():
        print("Error opening video stream or file")
        return
    
    if not out.isOpened():
        print("Error creating video writer")
        camera.release()
        return
    
    # Inference loop
    frame_count = 0
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            frame_count += 1
            
            # Ensure frame is in correct format (BGR, uint8)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            image = frame.copy()
            shape = image.shape[:2]
            
            r = args.inp_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]
            
            # Scale ratio (new / old)
            r = min(1.0, args.inp_size / height, args.inp_size / width)
            
            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = (args.inp_size - pad[0]) / 2
            h = (args.inp_size - pad[1]) / 2
            
            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
            
            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.to(device)
            
            # Convert to float32 for inference to avoid dtype mismatch
            x = x.float()
            x = x / 255
            
            # Inference with proper error handling
            try:
                outputs = model(x)
                if frame_count == 1:
                    print("First frame inference successful!")
            except Exception as e:
                print(f"Inference error on frame {frame_count}: {e}")
                print("Skipping frame due to inference error")
                continue
                    
            # NMS
            outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]
            
            if outputs is not None:
                outputs[:, [0, 2]] -= w
                outputs[:, [1, 3]] -= h
                outputs[:, :4] /= min(height / shape[0], width / shape[1])
                
                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])
                
                for box in outputs:
                    box = box.cpu().numpy()
                    score, index = box[4], box[5]
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    util.draw_box(frame, box, index, label)
            
            # Ensure frame is in correct format before writing
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            cv2.imshow('Frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Cleanup
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing completed successfully! Processed {frame_count} frames.")

def optimize_model(args, params):
    """
    Optimize the model using specified optimization techniques
    
    Args:
        args: Command line arguments
        params: Configuration parameters
    """
    print(f"Optimizing model with {args.opt_method}...")
    
    # Load model using smart loading
    if hasattr(args, 'weights') and args.weights:
        try:
            model = smart_load_model(args.weights, args.num_cls, device)
            print(f"Loaded weights from {args.weights}")
        except Exception as e:
            print(f"Error loading model with smart_load_model: {e}")
            # Fallback to creating new model
            model = nn.yolo_v11_n(args.num_cls).to(device)
            print("Created new model instead")
    else:
        model = nn.yolo_v11_n(args.num_cls).to(device)
    
    # Convert model to float and disable all Conv layer fusion BEFORE optimization
    model = model.float()
    
    def disable_conv_fusion(module):
        """Recursively disable fusion for all Conv modules"""
        for name, child in module.named_children():
            if hasattr(child, 'unfuse') and callable(child.unfuse):
                child.unfuse()
                print(f"Disabled fusion for Conv module: {name}")
            elif hasattr(child, '_use_fused'):
                child._use_fused = False
                print(f"Disabled fusion flag for module: {name}")
            # Recursively process child modules
            disable_conv_fusion(child)
    
    print("Disabling Conv layer fusion before optimization to avoid forward_fuse errors...")
    disable_conv_fusion(model)
    
    # For quantized models, calling eval() might cause issues
    try:
        model.eval()
        print("Model set to evaluation mode successfully")
    except Exception as e:
        print(f"Warning: Could not set model to eval mode: {e}")
        # Manually set training mode to False
        if hasattr(model, 'training'):
            model.training = False
            print("Manually set model training mode to False")
    
    # Apply optimization based on method
    if args.opt_method == 'quantization':
        optimizer = optimization.QuantizationOptimizer(model)
        
        if args.quant_type == 'dynamic':
            print("Applying dynamic quantization...")
            try:
                optimized_model = optimizer.dynamic_quantization()
            except RuntimeError as e:
                if "NoQEngine" in str(e):
                    print("Dynamic quantization failed due to missing quantization engine in your PyTorch installation.")
                    print("Consider reinstalling PyTorch with full quantization support, or try static quantization instead.")
                    return
                else:
                    raise e
        elif args.quant_type == 'static':
            print("Applying static quantization...")
            # For static quantization, we need to prepare and then convert with calibration
            try:
                optimizer.static_quantization_prepare(backend=backend)
                # Try to create a simple calibration dataset
                try:
                    # Create a dummy calibration dataset for basic calibration
                    print("Running calibration...")
                    # We'll create a simple calibration using dummy data
                    # In practice, you would use real data from your dataset
                    calibration_loader = None  # No calibration data for now
                    optimized_model = optimizer.static_quantization_convert(calibration_loader)
                except Exception as e:
                    print(f"Calibration failed, converting without calibration: {e}")
                    optimized_model = optimizer.static_quantization_convert()
            except RuntimeError as e:
                if "NoQEngine" in str(e):
                    print("Static quantization failed due to missing quantization engine in your PyTorch installation.")
                    print("Consider reinstalling PyTorch with full quantization support.")
                    return
                else:
                    raise e
        elif args.quant_type == 'qat':
            print("Preparing for quantization-aware training...")
            try:
                optimized_model = optimizer.quantization_aware_training_prepare(backend=backend)
            except RuntimeError as e:
                if "NoQEngine" in str(e):
                    print("Quantization-aware training preparation failed due to missing quantization engine in your PyTorch installation.")
                    print("Consider reinstalling PyTorch with full quantization support.")
                    return
                else:
                    raise e
        else:
            raise ValueError(f"Unknown quantization type: {args.quant_type}")
            
    elif args.opt_method == 'pruning':
        optimizer = optimization.PruningOptimizer(model)
        print(f"Applying pruning with sparsity {args.sparsity}...")
        optimized_model = optimizer.magnitude_pruning(sparsity=args.sparsity)
        
    elif args.opt_method == 'distillation':
        print("Distillation requires a teacher model. Please use the optimization module directly for this.")
        return
        
    else:
        raise ValueError(f"Unknown optimization method: {args.opt_method}")
    
    # Save optimized model
    os.makedirs(os.path.dirname(args.opt_output), exist_ok=True)
    if args.opt_method == 'quantization':
        optimizer.save_quantized_model(args.opt_output)
        print(f"Quantized model saved to {args.opt_output}")
    else:
        torch.save(optimized_model.state_dict(), args.opt_output)
        print(f"Optimized model saved to {args.opt_output}")
    
    # Compare model sizes
    try:
        size_info = optimization.compare_model_sizes(model, optimized_model)
        print("\nModel Size Comparison:")
        print(f"Original parameters: {size_info['original_params']:,}")
        print(f"Optimized parameters: {size_info['optimized_params']:,}")
        print(f"Compression ratio: {size_info['compression_ratio']:.2f}x")
        print(f"Memory savings: {size_info['memory_savings_mb']:.2f} MB")
    except Exception as e:
        print(f"Could not compare model sizes: {e}")

def main():
    """
    Main entry point for the YOLOv11 application
    Handles command line argument parsing and routes to appropriate functions
    """
    # Setup argument parser with comprehensive help
    parser = argparse.ArgumentParser(
        description="YOLOv11 Next-Gen Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:                  python main.py --train --epochs 100
  Resume training:              python main.py --train --resume weights/last.pt
  Validate model:               python main.py --validate
  Run inference:                python main.py --inference
  Evaluate model (full):        python main.py --evaluate
  Evaluate model (accuracy):    python main.py --evaluate --eval-mode accuracy
  Evaluate model (speed):       python main.py --evaluate --eval-mode speed
  Quick evaluation:             python main.py --evaluate --eval-mode quick
  Optimize model:               python main.py --optimize --opt-method quantization
        """
    )
    
    # Core arguments
    parser.add_argument('--rank', default=0, type=int,
                        help='Process rank for distributed training')
    parser.add_argument('--epochs', default=2, type=int,
                        help='Number of training epochs')
    parser.add_argument('--num-cls', type=int, default=80,
                        help='Number of object classes')
    parser.add_argument('--inp-size', type=int, default=640,
                        help='Input image size for training/inference')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='COCO',
                        help='Path to dataset directory')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run the model on (cpu, cuda, mps)')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['fbgemm', 'qnnpack'],
                        help='Backend for quantization (fbgemm or qnnpack)')

    # Mode selection flags
    parser.add_argument('--train', action='store_true',
                        help='Run training')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation')
    parser.add_argument('--inference', action='store_true',
                        help='Run inference on input.mp4')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation (accuracy and speed)')
    parser.add_argument('--optimize', action='store_true',
                        help='Run model optimization')
    
    # Evaluation arguments
    parser.add_argument('--eval-mode', type=str, default='full', 
                        choices=['full', 'accuracy', 'speed', 'quick'],
                        help='Evaluation mode: full, accuracy, speed, or quick')
    
    # Optimization arguments
    parser.add_argument('--opt-method', type=str, default='quantization',
                        choices=['quantization', 'pruning', 'distillation'],
                        help='Optimization method to apply')
    parser.add_argument('--opt-output', type=str, default='weights/optimized_model.pt',
                        help='Output path for optimized model')
    parser.add_argument('--quant-type', type=str, default='dynamic',
                        choices=['dynamic', 'static', 'qat'],
                        help='Type of quantization to apply')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='Sparsity level for pruning (0.0 to 1.0)')
    
    # Model and checkpoint arguments
    parser.add_argument('--weights', type=str, 
                        help='Path to pretrained weights file')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint file for resuming training')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup distributed training parameters
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    
    global device, backend
    if args.device is not None:
        device = torch.device(args.device)
    if args.backend is not None:
        backend = args.backend

    # Load configuration parameters
    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    
    # Route to appropriate function based on arguments
    if args.train:
        train(args, params)
    if args.validate:
        validate(args, params)
    if args.inference:
        inference(args, params)
    if args.evaluate:
        # Check if evaluation module is available
        if not EVAL_MODULE_AVAILABLE:
            print("Error: Evaluation module not found. Cannot run evaluation.")
            return
        
        # Import and run evaluation
        eval_args = argparse.Namespace()
        eval_args.num_cls = args.num_cls
        eval_args.inp_size = args.inp_size
        eval_args.batch_size = args.batch_size
        eval_args.data_dir = args.data_dir
        eval_args.mode = args.eval_mode
        
        evaluator = evaluation.Evaluator(eval_args, params)
        
        # Quick mode - reduced evaluation for faster results
        if args.eval_mode == 'quick':
            eval_args.num_images = 10  # Reduced number of images for speed test
        
        # Run evaluation
        evaluator.run_evaluation(mode=args.eval_mode if args.eval_mode != 'quick' else 'full')
        
        # Print report
        evaluator.print_report()
        
    if args.optimize:
        # Check if optimization module is available
        if not OPTIMIZATION_MODULE_AVAILABLE:
            print("Error: Optimization module not found. Cannot run optimization.")
            return
            
        optimize_model(args, params)

if __name__ == "__main__":
    main()