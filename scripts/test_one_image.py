import numpy as np
import cv2
import os
import imageio
from tqdm import tqdm


def create_visualization(mask_path, feature_path):
    """Create a single visualization frame"""
    mask = np.load(mask_path).astype(int)  # Shape: (L, H, W)
    features = np.load(feature_path)  # Shape: (N, feature_dim)
    
    l, h, w = mask.shape
    image = np.zeros((l, h, w, 3))
    
    for c in range(l):
        for i in range(h):
            for j in range(w):
                image[c,i,j] = features[mask[c,i,j], :3]  # Use first 3 feature dimensions
    
    return cv2.normalize(np.mean(image[2:3], axis=0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #return cv2.normalize(image[23], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def create_gif_from_directory(base_dir, output_gif="output.gif", fps=10):
    """Process directory and create GIF"""
    mask_dir = os.path.join(base_dir, "semantic_features")
    feature_dir = os.path.join(base_dir, "semantic_features-"+features+"_dim3")
    
    # Get all mask files and sort them numerically
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith('_s.npy')],
        key=lambda x: int(x.split('_')[0]))
    
    frames = []
    for mask_file in tqdm(mask_files, desc="Processing frames"):
        base_name = mask_file.split('_')[0]
        feature_file = f"{base_name}_f.npy"
        
        mask_path = os.path.join(mask_dir, mask_file)
        feature_path = os.path.join(feature_dir, feature_file)
        
        if not os.path.exists(feature_path):
            print(f"Warning: Missing feature file {feature_file} for {mask_file}")
            continue
            
        frame = create_visualization(mask_path, feature_path)
        frames.append(frame)
    
    # Save as GIF
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF saved to {output_gif} with {len(frames)} frames")



def create_orbit_gif(base_dir,out):
    frames = []
    for cam in tqdm(range(20)):
        mask = os.path.join(base_dir,str(cam), "semantic_features","50_s.npy")
        feat = os.path.join(base_dir,str(cam), "semantic_features-"+features+"_dim3", "50_f.npy")
        frame = create_visualization(mask, feat)
        frames.append(frame)

    # Save as GIF
    imageio.mimsave(out, frames, fps=5)




task = "sweep_to_dustpan_of_size"
data_directory = "/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/"+ task +"/all_variations/episodes/episode0/nerf_data_semantic"

for features in ["clip" , "dino", "combined" ]:

    create_gif_from_directory(data_directory+"/5", "viz_"+task+"/semantic_visualization_"+features+".gif", fps=10)
    create_orbit_gif(data_directory, "viz_"+task+"/orbit_"+features+".gif")