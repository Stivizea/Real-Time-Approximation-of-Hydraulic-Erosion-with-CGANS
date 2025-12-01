import numpy as np
import os
import pandas as pd
from numba import njit
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

# --- CONFIGURATION ---
INPUT_DIR = "terrain_dataset_10k"
OUTPUT_DIR = "terrain_dataset_eroded"
METADATA_IN = os.path.join(INPUT_DIR, "dataset_metadata.csv")
METADATA_OUT = os.path.join(OUTPUT_DIR, "eroded_metadata.csv")

# Simulation Constants
ITERATIONS = 200000  # Total drops per map
MAX_PATH_STEPS = 64  # Max steps a drop takes [cite: 358]

# --- NUMBA ACCELERATED PHYSICS KERNEL ---
# Based on the particle-based hydraulic erosion method described in Chapter 5 
# of the thesis (Beyer, 2015)[cite: 150, 151].

@njit(fastmath=True)
def erode_heightmap_jit(heightmap, iterations, p_inertia, p_gravity, p_capacity, p_evap, p_erosion, p_deposition, p_radius, p_min_slope):
    map_w, map_h = heightmap.shape
    
    for i in range(iterations):
        # 1. Spawn Drop at random position [cite: 151]
        pos_x = np.random.uniform(0, map_w - 2)
        pos_y = np.random.uniform(0, map_h - 2)
        dir_x = 0.0
        dir_y = 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0
        
        for step in range(MAX_PATH_STEPS):
            node_x = int(pos_x)
            node_y = int(pos_y)
            
            # Boundary Check
            if node_x < 0 or node_x >= map_w - 1 or node_y < 0 or node_y >= map_h - 1:
                break
                
            cell_offset_x = pos_x - node_x
            cell_offset_y = pos_y - node_y
            
            # 2. Calculate Gradient (Bilinear Interpolation) [cite: 160, 168]
            h00 = heightmap[node_x, node_y]
            h10 = heightmap[node_x + 1, node_y]
            h01 = heightmap[node_x, node_y + 1]
            h11 = heightmap[node_x + 1, node_y + 1]
            
            gx = (h10 - h00) * (1 - cell_offset_y) + (h11 - h01) * cell_offset_y
            gy = (h01 - h00) * (1 - cell_offset_x) + (h11 - h10) * cell_offset_x
            
            # 3. Compute new direction with Inertia [cite: 173]
            dir_x = (dir_x * p_inertia) - (gx * (1 - p_inertia))
            dir_y = (dir_y * p_inertia) - (gy * (1 - p_inertia))
            
            # Normalize direction
            len_dir = np.sqrt(dir_x**2 + dir_y**2)
            if len_dir != 0:
                dir_x /= len_dir
                dir_y /= len_dir
            else:
                dir_x = np.random.uniform(-1, 1)
                dir_y = np.random.uniform(-1, 1)
            
            # 4. Move Particle (Unit Step) [cite: 179]
            new_pos_x = pos_x + dir_x
            new_pos_y = pos_y + dir_y
            
            # Boundary Check for new position
            if new_pos_x < 0 or new_pos_x >= map_w - 1 or new_pos_y < 0 or new_pos_y >= map_h - 1:
                break
            
            # 5. Calculate Height Difference [cite: 192]
            n_node_x = int(new_pos_x)
            n_node_y = int(new_pos_y)
            n_off_x = new_pos_x - n_node_x
            n_off_y = new_pos_y - n_node_y
            
            nh00 = heightmap[n_node_x, n_node_y]
            nh10 = heightmap[n_node_x + 1, n_node_y]
            nh01 = heightmap[n_node_x, n_node_y + 1]
            nh11 = heightmap[n_node_x + 1, n_node_y + 1]
            
            new_height = nh00 * (1 - n_off_x) * (1 - n_off_y) + \
                         nh10 * n_off_x * (1 - n_off_y) + \
                         nh01 * (1 - n_off_x) * n_off_y + \
                         nh11 * n_off_x * n_off_y
            
            h_old = h00 * (1 - cell_offset_x) * (1 - cell_offset_y) + \
                    h10 * cell_offset_x * (1 - cell_offset_y) + \
                    h01 * (1 - cell_offset_x) * cell_offset_y + \
                    h11 * cell_offset_x * cell_offset_y

            diff = new_height - h_old
            
            # 6. Sediment Interaction
            if diff > 0:
                # Uphill movement: Deposit sediment to fill the pit [cite: 195]
                amount_to_deposit = min(diff, sediment)
                sediment -= amount_to_deposit
                
                heightmap[node_x, node_y] += amount_to_deposit * (1 - cell_offset_x) * (1 - cell_offset_y)
                heightmap[node_x + 1, node_y] += amount_to_deposit * cell_offset_x * (1 - cell_offset_y)
                heightmap[node_x, node_y + 1] += amount_to_deposit * (1 - cell_offset_x) * cell_offset_y
                heightmap[node_x + 1, node_y + 1] += amount_to_deposit * cell_offset_x * cell_offset_y
                
            else:
                # Downhill movement: Erode [cite: 197]
                # Capacity calculation (Eq 5.4) [cite: 202]
                capacity = max(-diff, p_min_slope) * speed * water * p_capacity
                
                if sediment > capacity:
                    # Drop surplus (Eq 5.5) [cite: 203, 306]
                    amount_to_deposit = (sediment - capacity) * p_deposition
                    sediment -= amount_to_deposit
                    
                    heightmap[node_x, node_y] += amount_to_deposit * (1 - cell_offset_x) * (1 - cell_offset_y)
                    heightmap[node_x + 1, node_y] += amount_to_deposit * cell_offset_x * (1 - cell_offset_y)
                    heightmap[node_x, node_y + 1] += amount_to_deposit * (1 - cell_offset_x) * cell_offset_y
                    heightmap[node_x + 1, node_y + 1] += amount_to_deposit * cell_offset_x * cell_offset_y
                
                else:
                    # Erode (Eq 5.6) [cite: 206]
                    amount_to_erode = min((capacity - sediment) * p_erosion, -diff)
                    
                    heightmap[node_x, node_y] -= amount_to_erode * (1 - cell_offset_x) * (1 - cell_offset_y)
                    heightmap[node_x + 1, node_y] -= amount_to_erode * cell_offset_x * (1 - cell_offset_y)
                    heightmap[node_x, node_y + 1] -= amount_to_erode * (1 - cell_offset_x) * cell_offset_y
                    heightmap[node_x + 1, node_y + 1] -= amount_to_erode * cell_offset_x * cell_offset_y
                    
                    sediment += amount_to_erode

            # 7. Update Velocity and Water
            # FIX: Prevent negative value in sqrt by using max(0, ...)
            # Physically: Speed increases when diff is negative (downhill).
            # Eq 5.7 [cite: 220]
            speed = np.sqrt(max(0.0, speed**2 - diff * p_gravity))
            water *= (1 - p_evap) # Eq 5.8 [cite: 221]
            
            pos_x = new_pos_x
            pos_y = new_pos_y
    
    return heightmap

# --- WRAPPER FOR MULTIPROCESSING ---

def process_single_erosion(args):
    """
    Worker function to handle one map: load, simulate, sanitize, save.
    """
    idx, input_path, raw_out, unity_out = args
    
    # Load and ensure float64 precision
    heightmap = np.load(input_path).astype(np.float64)
    
    # --- PARAMETER CONFIGURATION ---
    # Gravity Fixed [cite: 373, 375]
    p_gravity = 9.8 
    
    # Randomize other parameters for GAN robustness
    p_inertia = random.uniform(0.05, 0.3)      # [cite: 272]
    p_capacity = random.uniform(4.0, 8.0)      # [cite: 293]
    p_evap = random.uniform(0.01, 0.05)        # [cite: 326]
    p_erosion = random.uniform(0.1, 0.5)       # [cite: 315]
    p_deposition = random.uniform(0.1, 0.3)    # [cite: 306]
    p_radius = 4.0                             # [cite: 339]
    p_min_slope = 0.01                         # [cite: 352]

    # Run Simulation
    eroded_map = erode_heightmap_jit(
        heightmap, 
        ITERATIONS, 
        p_inertia, 
        p_gravity, 
        p_capacity, 
        p_evap, 
        p_erosion, 
        p_deposition, 
        p_radius,
        p_min_slope
    )
    
    # --- SANITIZATION ---
    # Fix for RuntimeWarning: Replace NaNs/Infs caused by physics spikes
    eroded_map = np.nan_to_num(eroded_map, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 1. Save Raw Scientific Data (.npy)
    np.save(raw_out, eroded_map)
    
    # 2. Save Unity Data (16-bit PNG)
    # Clip to valid range 0.0-1.0 to prevent overflow
    final_h_map = np.clip(eroded_map, 0.0, 1.0) 
    
    # Convert to 16-bit int
    uint16_data = (final_h_map * 65535).astype(np.uint16)
    img = Image.fromarray(uint16_data, mode='I;16')
    img.save(unity_out)
    
    return {
        "ID": idx,
        "p_inertia": p_inertia,
        "p_gravity": p_gravity,
        "p_capacity": p_capacity,
        "p_evaporation": p_evap,
        "p_erosion": p_erosion,
        "p_deposition": p_deposition
    }

def main():
    # Create output folders
    os.makedirs(os.path.join(OUTPUT_DIR, "npy_raw"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "unity_import"), exist_ok=True)
    
    print("Loading source dataset...")
    if not os.path.exists(METADATA_IN):
        print(f"Error: Could not find {METADATA_IN}. Run the generator script first.")
        return

    df_meta = pd.read_csv(METADATA_IN)
    
    # Prepare Tasks
    tasks = []
    for index, row in df_meta.iterrows():
        i = int(row['ID'])
        in_path = os.path.join(INPUT_DIR, "npy_raw", f"terrain_{i}.npy")
        raw_out = os.path.join(OUTPUT_DIR, "npy_raw", f"eroded_{i}.npy")
        unity_out = os.path.join(OUTPUT_DIR, "unity_import", f"eroded_{i}.png")
        tasks.append((i, in_path, raw_out, unity_out))
        
    print(f"Eroding {len(tasks)} maps using {cpu_count()} cores...")
    
    # Execute Parallel Processing
    erosion_metadata = []
    with Pool(processes=cpu_count()) as pool:
        # imap_unordered + tqdm gives a live progress bar
        for result in tqdm(pool.imap_unordered(process_single_erosion, tasks), total=len(tasks)):
            erosion_metadata.append(result)
            
    # Save Metadata
    print("Updating metadata...")
    df_erosion = pd.DataFrame(erosion_metadata)
    # Merge original params with erosion params
    final_df = pd.merge(df_meta, df_erosion, on="ID")
    
    final_df.to_csv(METADATA_OUT, index=False)
    print(f"Success. Metadata saved to {METADATA_OUT}")

if __name__ == "__main__":
    main()