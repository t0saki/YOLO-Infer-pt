#!/usr/bin/env python3
"""
Convert COCO segmentation labels to YOLO bounding box format
"""
import os
import numpy as np
from pathlib import Path

def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to bounding box"""
    coords = np.array(polygon_coords).reshape(-1, 2)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Convert to YOLO format (center_x, center_y, width, height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return center_x, center_y, width, height

def convert_segmentation_to_bbox(input_dir, output_dir):
    """Convert segmentation labels to bbox labels"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    processed = 0
    for label_file in input_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            bbox_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                polygon_coords = [float(x) for x in parts[1:]]
                
                # Convert polygon to bbox
                center_x, center_y, width, height = polygon_to_bbox(polygon_coords)
                
                # Format as YOLO bbox: class_id center_x center_y width height
                bbox_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                bbox_lines.append(bbox_line)
            
            # Write converted labels
            output_file = output_dir / label_file.name
            with open(output_file, 'w') as f:
                f.writelines(bbox_lines)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} files...")
                
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    print(f"Conversion complete! Processed {processed} files.")

if __name__ == "__main__":
    # Convert train labels
    print("Converting training labels...")
    convert_segmentation_to_bbox(
        "/Users/tosaki/dev/YOLO-Infer/yolo11_project/datasets/coco/labels/train2017",
        "./COCO/labels/train2017"
    )
    
    # Convert val labels  
    print("Converting validation labels...")
    convert_segmentation_to_bbox(
        "/Users/tosaki/dev/YOLO-Infer/yolo11_project/datasets/coco/labels/val2017",
        "./COCO/labels/val2017"
    )
    
    print("All done! You can now restart training.")