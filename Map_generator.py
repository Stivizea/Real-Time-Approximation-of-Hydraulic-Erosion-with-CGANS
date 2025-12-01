import numpy as np
import noise
import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import csv

# Configuration
WIDTH, HEIGHT = 512, 512
NUM_SAMPLES = 10000
OUTPUT_DIR = "terrain_dataset_10k"
RAW_DIR = os.path.join(OUTPUT_DIR, "npy_raw")
UNITY_DIR = os.path.join(OUTPUT_DIR, "unity_import")
METADATA_FILE = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

def generate_perlin_map(params):
    """
    Generates a map based on specific randomized parameters.
    """
    offset_x, offset_y, scale, octaves, persistence, lacunarity = params
    
    map_data = np.zeros((WIDTH, HEIGHT))
    
    for i in range(WIDTH):
        for j in range(HEIGHT):
            val = noise.pnoise2((i / scale) + offset_x, 
                                (j / scale) + offset_y, 
                                octaves=octaves, 
                                persistence=persistence, 
                                lacunarity=lacunarity, 
                                repeatx=1000000, 
                                repeaty=1000000, 
                                base=0)
            map_data[i][j] = val

    # Normalize to 0.0 - 1.0
    min_val = map_data.min()
    max_val = map_data.max()
    if max_val - min_val > 0:
        norm_map = (map_data - min_val) / (max_val - min_val)
    else:
        norm_map = map_data
    
    return norm_map

def save_for_unity(data, filename):
    uint16_data = (data * 65535).astype(np.uint16)
    img = Image.fromarray(uint16_data, mode='I;16')
    img.save(filename)

def process_single_map(args):
    """
    Worker function.
    args: (index, offset_x, offset_y, scale, octaves, persistence, lacunarity)
    """
    idx, off_x, off_y, scale, oct, pers, lac = args
    
    # 1. Generate
    # We pass the specific random params for this instance
    heightmap = generate_perlin_map((off_x, off_y, scale, oct, pers, lac))
    
    # 2. Save Scientific Data
    np.save(os.path.join(RAW_DIR, f"terrain_{idx}.npy"), heightmap)
    
    # 3. Save Visual Data
    save_for_unity(heightmap, os.path.join(UNITY_DIR, f"terrain_{idx}.png"))
    
    return f"terrain_{idx}" # Return ID to confirm success

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(UNITY_DIR, exist_ok=True)

    print(f"Preparing {NUM_SAMPLES} unique terrain configurations...")
    
    # Random State
    rng = np.random.RandomState(42)
    
    tasks = []
    metadata = []

    for i in range(NUM_SAMPLES):
        # --- DOMAIN RANDOMIZATION ---
        # We randomize the "character" of the terrain to make the GAN robust.
        
        # Scale: Zoom level (50 = jagged/tight, 150 = rolling/wide)
        scale = rng.uniform(50.0, 150.0)
        
        # Octaves: Level of detail (4 = smooth, 8 = very craggy)
        octaves = rng.randint(4, 9) 
        
        # Persistence: How much amplitude each octave contributes (roughness)
        persistence = rng.uniform(0.4, 0.6)
        
        # Lacunarity: Frequency gap between octaves
        lacunarity = rng.uniform(1.8, 2.2)
        
        # Offsets: Location in the noise world
        off_x = rng.uniform(0, 100000)
        off_y = rng.uniform(0, 100000)
        
        tasks.append((i, off_x, off_y, scale, octaves, persistence, lacunarity))
        
        # Store metadata for CSV
        metadata.append([i, scale, octaves, persistence, lacunarity])

    # Save Metadata Log
    with open(METADATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Scale", "Octaves", "Persistence", "Lacunarity"])
        writer.writerows(metadata)
    print(f"Metadata saved to {METADATA_FILE}")

    print(f"Starting parallel generation on {cpu_count()} cores...")
    
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_map, tasks), total=NUM_SAMPLES))

    print("Generation complete. 10,000 samples created.")

if __name__ == "__main__":
    main()