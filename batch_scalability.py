import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
from torch.cuda.amp import autocast

# --- IMPORTS ---
from Erotion_simulation import erode_heightmap_jit
from GAN_Model import Generator

# --- CONFIGURATION ---
CHECKPOINT_PATH = "gan_checkpoints/gen_epoch_100.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAP_SIZE = 512
# We test up to 64. If 64 still crashes, remove it and stop at 32.
BATCH_SIZES = [1, 4, 8, 16, 32, 64] 
ITERATIONS = 200000 

def benchmark():
    physics_times = []
    gan_times = []
    
    print("--- STARTING OPTIMIZED BENCHMARK ---")
    
    # 1. OPTIMIZATION: Enable cuDNN benchmarking
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load Model
    gen = Generator(in_channels=6).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        # weights_only=False fixes pickle warnings
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(checkpoint["state_dict"])
        gen.eval()
    
    # Warm up GPU (with FP16)
    print("  Warming up GPU...")
    dummy = torch.randn(1, 6, MAP_SIZE, MAP_SIZE).to(DEVICE)
    with autocast():
        with torch.no_grad():
            gen(dummy)
    torch.cuda.synchronize()

    for b_size in BATCH_SIZES:
        print(f"\nTesting Batch Size: {b_size}")
        
        # --- PREPARE DATA (OUTSIDE TIMER) ---
        # Generate dummy physics inputs
        phys_maps = [np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float64) for _ in range(b_size)]
        
        # Generate dummy GAN input (Random noise simulates loaded data)
        # We put it immediately on GPU to simulate "Ready-to-Infer" state
        gan_tensor = torch.randn(b_size, 6, MAP_SIZE, MAP_SIZE, dtype=torch.float32).to(DEVICE)
        
        # --- TEST PHYSICS (Sequential CPU) ---
        # Note: We use a smaller loop for the text output, but logic remains valid
        start_phys = time.perf_counter()
        for i in range(b_size):
            # We run the JIT function. 
            # Note: We pass standard params. 
            _ = erode_heightmap_jit(
                phys_maps[i], ITERATIONS, 0.1, 9.8, 8.0, 0.02, 0.3, 0.1, 4, 0.01
            )
        end_phys = time.perf_counter()
        p_time = end_phys - start_phys
        physics_times.append(p_time)
        print(f"  Physics Time: {p_time:.4f}s")
        
        # --- TEST GAN (Parallel GPU with AMP) ---
        # Clear VRAM before big batches
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()
        
        start_gan = time.perf_counter()
        
        # USE MIXED PRECISION (FP16) - Crucial for Batch 64
        with autocast():
            with torch.no_grad():
                _ = gen(gan_tensor)
                
        torch.cuda.synchronize() # Wait for GPU to finish
        end_gan = time.perf_counter()
        
        g_time = end_gan - start_gan
        gan_times.append(g_time)
        print(f"  GAN Time:     {g_time:.4f}s")

    return physics_times, gan_times

def plot_results(physics_times, gan_times):
    # Setup Data
    x = np.arange(len(BATCH_SIZES))
    width = 0.35
    
    # Professional Figure Setup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create Bars
    rects1 = ax.bar(x - width/2, physics_times, width, label='Physics Simulation (CPU)', color='#1f77b4', edgecolor='black', linewidth=0.5, alpha=0.9)
    rects2 = ax.bar(x + width/2, gan_times, width, label='cGAN Inference (GPU - FP16)', color='#2ca02c', edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # Aesthetics
    ax.set_ylabel('Total Execution Time (Seconds) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_xlabel('Batch Size (Number of Maps)', fontsize=12, fontweight='bold')
    ax.set_title('Scalability Comparison: Physics vs. Deep Learning Surrogate', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(BATCH_SIZES, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    
    # Log Scale
    ax.set_yscale('log')
    # Add a light grid for readability
    ax.grid(True, which="major", axis='y', linestyle='-', alpha=0.3)
    ax.grid(True, which="minor", axis='y', linestyle=':', alpha=0.2)
    
    # Calculate Speedup Factors
    speedups = [p / g for p, g in zip(physics_times, gan_times)]
    
    # Annotate Speedups
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        speedup = speedups[i]
        
        # Dynamic positioning for text
        text_y = height * 1.2 # slightly above bar
        
        ax.annotate(f'{speedup:.0f}x',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), 
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='#2ca02c')

    # Draw trend lines to visualize the divergence
    ax.plot(x - width/2, physics_times, 'o--', color='#1f4e79', alpha=0.4, linewidth=1)
    ax.plot(x + width/2, gan_times, 'o--', color='#1e701e', alpha=0.4, linewidth=1)

    plt.tight_layout()
    output_file = "Figure_3_Scalability_Final.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nChart saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    p_times, g_times = benchmark()
    plot_results(p_times, g_times)