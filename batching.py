import os
import argparse
import numpy as np
import open3d as o3d
import torch
import glob
import struct
from scipy.spatial.transform import Rotation
import sys
from PIL import Image
import cv2
import requests
import tempfile

from collections import defaultdict

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    return model, device

def merge_prediction_dicts(dict1, dict2):
    merged = {}

    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        # Handle numpy arrays or torch tensors
        if hasattr(val1, 'shape') and hasattr(val2, 'shape'):
            try:
                merged[key] = val1.__class__(np.concatenate([val1, val2], axis=0))
            except Exception:
                print("Fallback")
                merged[key] = val1  # fallback
        # Handle lists
        elif isinstance(val1, list) and isinstance(val2, list):
            merged[key] = val1 + val2
        # Fallback: keep the first or raise an error
        else:
            raise ValueError(f"Cannot merge key '{key}' of type {type(val1)}")

    return merged

def init_template_dict(reference_dict):
    template = {}
    for key, value in reference_dict.items():
        if isinstance(value, list):
            template[key] = []
        elif isinstance(value, np.ndarray):
            template[key] = np.empty((0, *value.shape[1:]), dtype=value.dtype)
        elif isinstance(value, torch.Tensor):
            template[key] = torch.empty((0, *value.shape[1:]), dtype=value.dtype, device=value.device)
        else:
            # fallback: set to None or empty list
            template[key] = None
    return template

def process_images(image_dir, model, device,fraction = 1.0):
    """Process images with VGGT and return predictions."""
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    image_names = image_names[:int(fraction * len(image_names))]
    print(f"Found {len(image_names)} images")
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    original_images = []
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("Running inference...")
    point_map = None
    point_conf = None
    point_colors = None
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    for idx in range(0,len(images),10):
        sub_images = images[idx:idx + 10]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                sub_images = sub_images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(sub_images)        
                sub_point_map, sub_point_conf = model.point_head(aggregated_tokens_list, sub_images, ps_idx)
        if(point_map is None):
            point_map = sub_point_map
            point_conf = sub_point_conf
            point_colors = sub_images
        else:
            point_map = torch.concat((point_map,sub_point_map),dim = 1)
            point_conf = torch.concat((point_conf,sub_point_conf), dim = 1)
            point_colors = torch.concat((point_colors,sub_images),dim = 1)

    return point_map,point_conf,point_colors


def filter_and_save_points(points,conf, colors_rgb, output_dir = 'VGGT2COL'):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    num_images = points.shape[1]
    conf_threshold = (num_images / 40) * 20

    vertices_3d = points.reshape(-1, 3).cpu().numpy()
    conf = conf.reshape(-1).cpu().numpy()
    colors_rgb_flat = colors_rgb.reshape(-1, 3).cpu().numpy()
    

    if len(conf) != len(colors_rgb_flat):
        print(f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})")
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]
    
    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)
    
    print(f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})")
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)
    
    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]
    filtered_conf = conf[conf_mask]
    
    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])


    # Make sure filtered_points and filtered_colors are numpy arrays
    # filtered_points: Nx3 array
    # filtered_colors: Nx3 array (RGB values in range [0, 1])   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)

    # Apply voxel downsampling
    voxel_size = 0.005  # change as needed
    # downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    voxel_coords = np.floor(filtered_vertices / voxel_size).astype(np.int32)

    # Step 3: Group indices by voxel
    voxel_dict = defaultdict(list)
    for idx, voxel in enumerate(voxel_coords):
        voxel_key = tuple(voxel)
        voxel_dict[voxel_key].append(idx)

    # Step 4: Downsample using Open3D
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downpcd.points)

    # # # # Convert back to NumPy arrays
    filtered_vertices = np.asarray(downpcd.points)
    filtered_colors = (np.asarray(downpcd.colors) * 255).astype(np.uint8)

    # # Save as .ply
    # print("Saving PointCloud")
    o3d.io.write_point_cloud(os.path.join(output_dir,f"Point_{int(conf_threshold)}.ply"), pcd)
    o3d.io.write_point_cloud(os.path.join(output_dir,f"Point_Voxel_{int(conf_threshold)}.ply"),downpcd)

    print(f"Filtered to {len(filtered_vertices)} points")


def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="colmap_output", 
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (0-100%) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--binary", action="store_true", 
                        help="Output binary COLMAP files instead of text")
    parser.add_argument("--stride", type=int, default=1, 
                        help="Stride for point sampling (higher = fewer points)")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which prediction branch to use")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="What fraction of the total images to use.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, device = load_model()
    
    point_map, point_conf, point_colors = process_images(args.image_dir, model, device,args.fraction)
    
    filter_and_save_points(point_map,point_conf,point_colors,args.output_dir)

if __name__ == "__main__":
    main()