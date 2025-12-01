import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import noise
import random
import os

# --- IMPORTS FROM YOUR FILES ---
# We import the physics engine and the Neural Network class
from Erotion_simulation import erode_heightmap_jit
from GAN_Model import Generator

# --- CONFIGURATION ---
CHECKPOINT_PATH = "gan_checkpoints/gen_epoch_100.pth.tar" # Path to your final model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP_SIZE = 512

def generate_random_perlin():
    """Generates a fresh 512x512 map using the same random logic as training."""
    # Randomize terrain parameters (Domain Randomization)
    scale = random.uniform(50.0, 150.0)
    octaves = random.randint(4, 9)
    persistence = random.uniform(0.4, 0.6)
    lacunarity = random.uniform(1.8, 2.2)
    offset_x = random.uniform(0, 100000)
    offset_y = random.uniform(0, 100000)

    map_data = np.zeros((MAP_SIZE, MAP_SIZE))
    
    # Generate Noise
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            val = noise.pnoise2((i / scale) + offset_x, 
                                (j / scale) + offset_y, 
                                octaves=octaves, 
                                persistence=persistence, 
                                lacunarity=lacunarity, 
                                repeatx=1000000, 
                                repeaty=1000000, 
                                base=0)
            map_data[i][j] = val

    # Normalize 0.0 to 1.0 (Required for Physics Engine)
    min_val = map_data.min()
    max_val = map_data.max()
    if max_val - min_val > 0:
        norm_map = (map_data - min_val) / (max_val - min_val)
    else:
        norm_map = map_data
        
    return norm_map

def run_comparison():
    print("--- 1. Generating Random Terrain ---")
    input_map = generate_random_perlin()
    
    # Define Simulation Parameters (The "Recipe")
    # We pick random values to test if the GAN generalizes
    params = {
        'p_inertia': random.uniform(0.05, 0.3),
        'p_gravity': 9.8, # Fixed
        'p_capacity': random.uniform(4.0, 8.0),
        'p_evap': random.uniform(0.01, 0.05),
        'p_erosion': random.uniform(0.1, 0.5),
        'p_deposition': random.uniform(0.1, 0.3),
        'p_radius': 4,
        'p_min_slope': 0.01,
        'iterations': 200000
    }
    
    print(f"Parameters: Inertia={params['p_inertia']:.2f}, Erosion={params['p_erosion']:.2f}, Deposition={params['p_deposition']:.2f}")

    # ==========================================
    # METHOD A: PHYSICS SIMULATION (Ground Truth)
    # ==========================================
    print("\n--- 2. Running Physics Simulation (Ground Truth) ---")
    start_phys = time.perf_counter()
    
    # Copy map because simulation modifies it in-place
    map_for_physics = input_map.copy().astype(np.float64)
    
    # Call the JIT compiled function
    eroded_physics = erode_heightmap_jit(
        map_for_physics, 
        params['iterations'],
        params['p_inertia'],
        params['p_gravity'],
        params['p_capacity'],
        params['p_evap'],
        params['p_erosion'],
        params['p_deposition'],
        params['p_radius'],
        params['p_min_slope']
    )
    
    end_phys = time.perf_counter()
    time_physics = end_phys - start_phys
    print(f"Physics Simulation finished in: {time_physics:.4f} seconds")

    # ==========================================
    # METHOD B: cGAN PREDICTION (AI)
    # ==========================================
    print("\n--- 3. Running cGAN Inference ---")
    
    # 1. Load Model
    gen = Generator(in_channels=6).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(checkpoint["state_dict"])
        gen.eval()
    else:
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 2. Prepare Input Tensor
    # Normalize Input Map to [-1, 1] for GAN (Matches training logic)
    input_tensor_map = (input_map.astype(np.float32) * 2.0) - 1.0
    
    # Create Parameter Channels (Broadcasting scalars to 512x512)
    # Order matches 'dataset.param_cols' from training: 
    # ['p_inertia', 'p_capacity', 'p_evaporation', 'p_erosion', 'p_deposition']
    param_list = [
        params['p_inertia'], 
        params['p_capacity'], 
        params['p_evap'], 
        params['p_erosion'], 
        params['p_deposition']
    ]
    
    h, w = MAP_SIZE, MAP_SIZE
    param_channels = []
    for p in param_list:
        # Fill channel with value p
        channel = np.full((h, w), p, dtype=np.float32)
        param_channels.append(channel)
        
    # Stack: [Input, Param1, Param2, ..., Param5] -> Shape (6, 512, 512)
    combined_input = np.stack([input_tensor_map] + param_channels, axis=0)
    
    # Add Batch Dimension -> (1, 6, 512, 512) and move to GPU
    tensor_input = torch.from_numpy(combined_input).unsqueeze(0).to(DEVICE)

    # 3. Run Inference & Time it
    start_gan = time.perf_counter()
    
    with torch.no_grad():
        prediction = gen(tensor_input)
        
    end_gan = time.perf_counter()
    time_gan = end_gan - start_gan
    print(f"cGAN Inference finished in: {time_gan:.4f} seconds")
    
    # 4. Post-process (Denormalize [-1, 1] -> [0, 1])
    eroded_gan = (prediction[0, 0].cpu().numpy() + 1) / 2
    
    # Clip just to be safe for display
    eroded_gan = np.clip(eroded_gan, 0, 1)

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\n--- 4. Visualizing Results ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original
    im1 = axes[0].imshow(input_map, cmap='gray')
    axes[0].set_title("Original Random Input\n(Perlin Noise)", fontsize=14)
    axes[0].axis('off')
    
    # Plot 2: Physics
    im2 = axes[1].imshow(eroded_physics, cmap='gray')
    axes[1].set_title(f"Ground Truth (Physics)\nTime: {time_physics:.4f} s", fontsize=14, color='black')
    axes[1].axis('off')
    
    # Plot 3: GAN
    im3 = axes[2].imshow(eroded_gan, cmap='gray')
    
    # Calculate Speedup
    speedup = time_physics / time_gan if time_gan > 0 else 0
    
    axes[2].set_title(f"cGAN Prediction (AI)\nTime: {time_gan:.4f} s ", fontsize=14, color='black')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_file = "comparison_result.png"
    plt.savefig(output_file)
    print(f"Comparison saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    run_comparison()