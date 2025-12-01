import numpy as np
import matplotlib.pyplot as plt
import noise
import random
import time

# --- IMPORT YOUR SIMULATION ---
# This assumes Erotion_simulation.py is in the same folder
from Erotion_simulation import erode_heightmap_jit

# --- CONFIGURATION ---
MAP_SIZE = 512
OUTPUT_FILE = "Figure_2_Physical_Erosion.png"
DPI = 300

def generate_perlin_map(seed=None):
    """Generates a base terrain for the demonstration."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        
    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
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

    # Normalize 0-1
    min_val = map_data.min()
    max_val = map_data.max()
    norm_map = (map_data - min_val) / (max_val - min_val)
    return norm_map

def create_figure():
    print("Generating base terrain...")
    # We use a fixed seed here so the figure is reproducible every time you run it
    # You can remove 'seed=42' to get random maps
    input_map = generate_perlin_map(seed=42)
    
    # --- SIMULATION PARAMETERS ---
    # These parameters are chosen to make the erosion very visible for the paper
    params = {
        'iterations': 300000,   # High iteration count for clear ravines
        'p_inertia': 0.1,       # Low inertia = water follows curves (winding rivers)
        'p_gravity': 9.8,
        'p_capacity': 8.0,      # High capacity = moves a lot of soil
        'p_evap': 0.02,
        'p_erosion': 0.3,       # Moderate erosion hardness
        'p_deposition': 0.1,    # Low deposition to avoid spikes
        'p_radius': 4,
        'p_min_slope': 0.01
    }

    print("Running hydraulic erosion simulation (Physics)...")
    start_time = time.perf_counter()
    
    # Prepare data for Numba (float64 copy)
    map_to_erode = input_map.copy().astype(np.float64)
    
    eroded_map = erode_heightmap_jit(
        map_to_erode,
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
    
    duration = time.perf_counter() - start_time
    print(f"Simulation complete in {duration:.2f} seconds.")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Original Map
    im1 = axes[0].imshow(input_map, cmap='gray')
    axes[0].set_title("(a) Initial Procedural Terrain (fBm)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Eroded Map
    im2 = axes[1].imshow(eroded_map, cmap='gray')
    axes[1].set_title("(b) Hydraulically Eroded Terrain (Ground Truth)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add Parameter Text Box
    # This creates a scientific looking box at the bottom
    param_text = (
        f"Simulation Parameters:\n"
        f"Iterations (N): {params['iterations']:,}\n"
        f"Inertia ($p_{{inertia}}$): {params['p_inertia']}\n"
        f"Gravity ($p_{{gravity}}$): {params['p_gravity']}\n"
        f"Sediment Capacity ($p_{{capacity}}$): {params['p_capacity']}\n"
        f"Evaporation Rate ($p_{{evap}}$): {params['p_evap']}\n"
        f"Erosion Rate ($p_{{erosion}}$): {params['p_erosion']}\n"
        f"Deposition Rate ($p_{{deposition}}$): {params['p_deposition']}"
    )
    
    # Place text in the figure
    fig.text(0.5, 0.15, param_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    # Adjust layout to make room for text
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    create_figure()