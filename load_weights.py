import torch
import os
from nets import nn
from utils.util import load_ultralytics_weight

def load_weights(num_classes=80, verbose=False):
    """Load model weights with priority: Ultralytics weights override our weights if available"""
    
    # Create model
    model = nn.yolo_v11_n(num_classes)
    if verbose:
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load our model weights first
    our_weights_path = 'weights_old/v11_n.pt'
    if os.path.exists(our_weights_path):
        try:
            checkpoint = torch.load(our_weights_path, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                # Handle case where checkpoint['model'] is already a model object
                if hasattr(checkpoint['model'], 'state_dict'):
                    model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
                else:
                    model.load_state_dict(checkpoint['model'], strict=False)
                if verbose:
                    print(f"Successfully loaded our weights from {our_weights_path}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                if verbose:
                    print(f"Successfully loaded our weights from {our_weights_path}")
        except Exception as e:
            if verbose:
                print(f"Error loading our weights: {e}")
    else:
        if verbose:
            print(f"Our weights file {our_weights_path} not found, using initialized weights")
    
    # Try to load Ultralytics weights if available
    ultralytics_weights_path = 'yolo11n.pt'
    if os.path.exists(ultralytics_weights_path):
        try:
            model = load_ultralytics_weight(model, ultralytics_weights_path)
            if verbose:
                print(f"Successfully loaded Ultralytics weights from {ultralytics_weights_path}")
                print("Ultralytics weights have overridden matching layers")
        except Exception as e:
            if verbose:
                print(f"Error loading Ultralytics weights: {e}")
                print("Keeping our weights for all layers")
    else:
        if verbose:
            print(f"Ultralytics weights file {ultralytics_weights_path} not found")
            print("Keeping our weights for all layers")
    
    return model

def save_weights(model, save_path='weights/combined_weights.pt'):
    """Save model weights to a .pt file in the same format as weights_old/v11_n.pt"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save in the same format as weights_old/v11_n.pt
    # The v11_n.pt file stores the model object directly, not just the state_dict
    torch.save({
        'model': model
    }, save_path)
    
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    model = load_weights()
    print("Weight loading completed!")
    
    # Save the combined weights
    save_weights(model, 'weights/combined_weights.pt')
    
    # Show some model info
    print("\nModel state dict keys (first 10):")
    keys = list(model.state_dict().keys())
    for i, key in enumerate(keys[:10]):
        print(f"  {key}")
    if len(keys) > 10:
        print(f"  ... and {len(keys) - 10} more keys")