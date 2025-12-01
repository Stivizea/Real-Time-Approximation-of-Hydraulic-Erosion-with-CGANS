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
BATCH_SIZE = 16  # Process 16 maps at once to show GPU power

def generate_random_perlin():
    """Generates a fresh 512x512 map."""
    scale = random.uniform(50.0, 150.0)
    octaves = random.randint(4, 9)
    persistence = random.uniform(0.4, 0.6)
    lacunarity = random.uniform(1.8, 2.2)
    offset_x = random.uniform(0, 100000)
    offset_y = random.uniform(0, 100000)

    map_data = np.zeros((MAP_SIZE, MAP_SIZE))
    
    # Simple unoptimized perlin loop (CPU)
    # We don't time this part, as we are comparing Erosion speeds only.
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

def run_batch_comparison():
    print(f"--- Preparing Batch of {BATCH_SIZE} Maps ---")
    
    # 1. Generate Input Data
    input_maps = []
    param_list = []
    
    for _ in range(BATCH_SIZE):
        input_maps.append(generate_random_perlin())
        
        # Unique parameters for each map in the batch
        p = {
            'p_inertia': random.uniform(0.05, 0.3),
            'p_gravity': 9.8,
            'p_capacity': random.uniform(4.0, 8.0),
            'p_evap': random.uniform(0.01, 0.05),
            'p_erosion': random.uniform(0.1, 0.5),
            'p_deposition': random.uniform(0.1, 0.3),
            'p_radius': 4,
            'p_min_slope': 0.01,
            'iterations': 200000
        }
        param_list.append(p)

    print("Data generated. Starting competition...")

    # ==========================================
    # METHOD A: PHYSICS SIMULATION (Sequential)
    # ==========================================
    print(f"\n--- Running Physics Simulation on {BATCH_SIZE} maps (Sequential CPU) ---")
    start_phys = time.perf_counter()
    
    physics_results = []
    
    # CPU must loop through them one by one
    for i in range(BATCH_SIZE):
        map_copy = input_maps[i].copy().astype(np.float64)
        params = param_list[i]
        
        eroded = erode_heightmap_jit(
            map_copy, 
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
        physics_results.append(eroded)
        # Optional: Print dot to show progress
        print(".", end="", flush=True)
    
    end_phys = time.perf_counter()
    total_time_phys = end_phys - start_phys
    print(f"\nPhysics finished in: {total_time_phys:.4f} seconds")
    print(f"Average time per map: {total_time_phys/BATCH_SIZE:.4f} seconds")

    # ==========================================
    # METHOD B: cGAN PREDICTION (Parallel Batch)
    # ==========================================
    print(f"\n--- Running cGAN Inference on {BATCH_SIZE} maps (Parallel GPU) ---")
    
    # 1. Load Model
    gen = Generator(in_channels=6).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(checkpoint["state_dict"])
        gen.eval()
    else:
        print("Model not found.")
        return

    # 2. Prepare Batch Tensor (The expensive setup part)
    tensor_batch_list = []
    
    for i in range(BATCH_SIZE):
        # Normalize map
        in_map = (input_maps[i].astype(np.float32) * 2.0) - 1.0
        p = param_list[i]
        
        # Create parameter channels
        p_vals = [p['p_inertia'], p['p_capacity'], p['p_evap'], p['p_erosion'], p['p_deposition']]
        channels = [in_map]
        
        for val in p_vals:
            channels.append(np.full((MAP_SIZE, MAP_SIZE), val, dtype=np.float32))
            
        # Stack channels [6, 512, 512]
        combined = np.stack(channels, axis=0)
        tensor_batch_list.append(torch.from_numpy(combined))
    
    # Stack into one big batch [BATCH_SIZE, 6, 512, 512]
    # Pin memory for faster transfer
    batch_tensor = torch.stack(tensor_batch_list).to(DEVICE)
    
    # 3. GPU WARM-UP (Crucial for accurate timing)
    # The first run always includes CUDA initialization overhead. We skip it.
    print("Warming up GPU...")
    with torch.no_grad():
        _ = gen(batch_tensor[0:1]) # Run 1 sample
    torch.cuda.synchronize() # Wait for warm-up to finish

    # 4. Run Inference & Time it
    print("Executing Batch Inference...")
    start_gan = time.perf_counter()
    
    with torch.no_grad():
        # THIS IS THE MAGIC: We process ALL 16 maps in one function call
        predictions = gen(batch_tensor)
        
    # Wait for all GPU kernels to finish before stopping timer
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    end_gan = time.perf_counter()
    total_time_gan = end_gan - start_gan
    
    print(f"cGAN Batch finished in: {total_time_gan:.4f} seconds")
    print(f"Average time per map: {total_time_gan/BATCH_SIZE:.4f} seconds")

    # ==========================================
    # REPORT
    # ==========================================
    speedup = total_time_phys / total_time_gan
    print(f"\n=============================================")
    print(f"FINAL RESULT (Batch Size: {BATCH_SIZE})")
    print(f"Physics Total Time: {total_time_phys:.4f} s")
    print(f"AI Total Time:      {total_time_gan:.4f} s")
    print(f"SPEEDUP FACTOR:     {speedup:.2f}x FASTER")
    print(f"=============================================")
    
    # ==========================================
    # VISUALIZATION (Just show the first one)
    # ==========================================
    print("\nVisualizing the first map of the batch...")
    
    gan_out_first = (predictions[0, 0].cpu().numpy() + 1) / 2
    gan_out_first = np.clip(gan_out_first, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(input_maps[0], cmap='gray')
    axes[0].set_title("Input (Map #1 of 16)")
    axes[0].axis('off')
    
    axes[1].imshow(physics_results[0], cmap='gray')
    axes[1].set_title("Physics Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(gan_out_first, cmap='gray')
    axes[2].set_title(f"cGAN Prediction\n(Batch Speedup: {speedup:.0f}x)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("batch_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_batch_comparison()