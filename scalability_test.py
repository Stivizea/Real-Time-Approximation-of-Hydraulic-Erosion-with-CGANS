import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import noise
import random
import os

# --- IMPORTS ---
from Erotion_simulation import erode_heightmap_jit
from GAN_Model import Generator

# --- CONFIGURATION ---
CHECKPOINT_PATH = "gan_checkpoints/gen_epoch_100.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP_SIZE = 512
STEPS = 5
DROPS_PER_STEP = 200000

def generate_random_perlin():
    """Generates a fresh 512x512 map."""
    scale = random.uniform(50.0, 150.0)
    octaves = random.randint(4, 9)
    persistence = random.uniform(0.4, 0.6)
    lacunarity = random.uniform(1.8, 2.2)
    offset_x = random.uniform(0, 100000)
    offset_y = random.uniform(0, 100000)

    map_data = np.zeros((MAP_SIZE, MAP_SIZE))
    
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

    min_val = map_data.min()
    max_val = map_data.max()
    if max_val - min_val > 0:
        norm_map = (map_data - min_val) / (max_val - min_val)
    else:
        norm_map = map_data
        
    return norm_map

def run_scalability_test():
    print("--- 1. Initialization ---")
    base_map = generate_random_perlin()
    
    # Random Physics Parameters
    params = {
        'p_inertia': random.uniform(0.05, 0.3),
        'p_gravity': 9.8,
        'p_capacity': random.uniform(4.0, 8.0),
        'p_evap': random.uniform(0.01, 0.05),
        'p_erosion': random.uniform(0.1, 0.5),
        'p_deposition': random.uniform(0.1, 0.3),
        'p_radius': 4,
        'p_min_slope': 0.01,
        'iterations': DROPS_PER_STEP 
    }
    
    print(f"Testing Scalability up to {STEPS * DROPS_PER_STEP:,} drops.")
    
    # ==========================================
    # METHOD A: CUMULATIVE PHYSICS
    # ==========================================
    print("\n--- 2. Running Cumulative Physics ---")
    physics_snapshots = []
    
    # Create a copy we will modify in-place
    current_phys_map = base_map.copy().astype(np.float64)
    start_phys = time.perf_counter()
    
    for i in range(STEPS):
        print(f"Physics Step {i+1}/{STEPS} (+200k drops)...")
        current_phys_map = erode_heightmap_jit(
            current_phys_map, 
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
        physics_snapshots.append(current_phys_map.copy())
        
    total_phys_time = time.perf_counter() - start_phys

    # ==========================================
    # METHOD B: RECURSIVE GAN
    # ==========================================
    print("\n--- 3. Running Recursive cGAN ---")
    
    gen = Generator(in_channels=6).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(checkpoint["state_dict"])
        gen.eval()
    else:
        print("Model not found.")
        return

    gan_snapshots = []
    
    # Prepare Parameters
    param_list = [params['p_inertia'], params['p_capacity'], params['p_evap'], params['p_erosion'], params['p_deposition']]
    param_channels = []
    for p in param_list:
        param_channels.append(np.full((MAP_SIZE, MAP_SIZE), p, dtype=np.float32))
    
    # Prepare Initial Input (Normalized -1 to 1)
    current_gan_input = (base_map.astype(np.float32) * 2.0) - 1.0
    
    # Ensure 4 Dimensions [Batch, Channel, Height, Width]
    current_gan_tensor = torch.from_numpy(current_gan_input).unsqueeze(0).unsqueeze(0).to(DEVICE) 
    param_tensor_stack = torch.from_numpy(np.stack(param_channels)).unsqueeze(0).to(DEVICE)

    start_gan = time.perf_counter()

    with torch.no_grad():
        for i in range(STEPS):
            print(f"GAN Recursive Pass {i+1}/{STEPS}...")
            
            combined_input = torch.cat([current_gan_tensor, param_tensor_stack], dim=1) 
            prediction = gen(combined_input)
            
            # Store Result (Denormalize for viewing)
            gan_out_view = (prediction[0, 0].cpu().numpy() + 1) / 2
            gan_snapshots.append(np.clip(gan_out_view, 0, 1))
            
            # Recursion
            current_gan_tensor = prediction
            
    total_gan_time = time.perf_counter() - start_gan

    # ==========================================
    # CALCULATE ERROR METRICS
    # ==========================================
    print("\n--- 4. Calculating Errors ---")
    error_maps = []
    mse_values = []
    
    for i in range(STEPS):
        p_map = physics_snapshots[i]
        g_map = gan_snapshots[i]
        
        # Absolute Difference (0.0 to 1.0)
        diff = np.abs(p_map - g_map)
        error_maps.append(diff)
        
        # Mean Squared Error
        mse = np.mean((p_map - g_map) ** 2)
        mse_values.append(mse)

    # Determine global max error for consistent heatmap scaling
    global_max_error = max([np.max(m) for m in error_maps])
    print(f"Global Max Pixel Error (for heatmap scale): {global_max_error:.4f}")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("--- 5. Generating Figure ---")
    fig, axes = plt.subplots(STEPS, 3, figsize=(15, 4 * STEPS))
    
    # Main Title
    fig.suptitle(f"Recursive Scalability Analysis: Physics vs. cGAN ({STEPS * DROPS_PER_STEP} Drops)", fontsize=20, fontweight='bold', y=0.96)
    
    # Column Headers (Only on top row)
    axes[0, 0].set_title(f"Physics Simulation (Ground Truth)\nTotal Time: {total_phys_time:.2f}s", fontsize=14, fontweight='bold', color='#1f77b4')
    axes[0, 1].set_title(f"Recursive cGAN (AI)\nTotal Time: {total_gan_time:.2f}s", fontsize=14, fontweight='bold', color='#2ca02c')
    axes[0, 2].set_title(f"Pixel Error Heatmap (|Phys - AI|)\nScale: 0 to {global_max_error*100:.1f}%", fontsize=14, fontweight='bold', color='#d62728')

    for i in range(STEPS):
        drop_count = (i + 1) * DROPS_PER_STEP
        
        # 1. Physics Map
        axes[i, 0].imshow(physics_snapshots[i], cmap='gray')
        axes[i, 0].set_ylabel(f"Step {i+1}\n({drop_count:,} Drops)", fontsize=12, rotation=0, labelpad=50, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # 2. GAN Map
        axes[i, 1].imshow(gan_snapshots[i], cmap='gray')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        
        # 3. Error Heatmap
        # Use 'inferno' or 'magma' for error visibility. 
        # vmin=0, vmax=global_max_error ensures consistent color across rows
        im = axes[i, 2].imshow(error_maps[i], cmap='inferno', vmin=0, vmax=global_max_error)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        
        # Add MSE Text Overlay
        axes[i, 2].text(
            10, 30, 
            f"MSE: {mse_values[i]:.2e}", 
            color='white', 
            fontsize=12, 
            fontweight='bold', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    # Add a colorbar for the heatmap
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Absolute Error Magnitude (0.0 - 1.0)')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1)
    
    output_file = "Figure_Recursive_Error_Analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"Scalability chart saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    run_scalability_test()