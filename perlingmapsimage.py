import numpy as np
import matplotlib.pyplot as plt
import noise
import random

# --- CONFIGURATION ---
MAP_SIZE = 512
DPI = 300
OUTPUT_FILE = "Figure_1_Perlin_Samples.png"

def generate_perlin_map(scale, octaves, persistence, lacunarity, seed_offset):
    """
    Generates a 512x512 Perlin noise map with specific parameters.
    """
    map_data = np.zeros((MAP_SIZE, MAP_SIZE))
    
    # We use a fixed offset per map to ensure reproducibility for the figure
    offset_x = seed_offset
    offset_y = seed_offset

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

    # Normalize to 0.0 - 1.0 for visualization
    min_val = map_data.min()
    max_val = map_data.max()
    if max_val - min_val > 0:
        norm_map = (map_data - min_val) / (max_val - min_val)
    else:
        norm_map = map_data
        
    return norm_map

def create_figure():
    # Define 4 distinct parameter sets to illustrate the algorithm's range
    # Case A: Standard Balanced Terrain
    # Case B: Smooth, Rolling Hills (Low Octaves, Low Persistence)
    # Case C: Rough, Jagged Mountains (High Octaves, High Persistence)
    # Case D: High Frequency/Zoomed Out (Low Scale)
    
    samples = [
        {
            "label": "(a) Balanced Terrain",
            "scale": 100.0, "octaves": 6, "persistence": 0.5, "lacunarity": 2.0, "seed": 1000
        },
        {
            "label": "(b) Smooth / Rolling",
            "scale": 150.0, "octaves": 3, "persistence": 0.4, "lacunarity": 1.8, "seed": 2000
        },
        {
            "label": "(c) Rough / Jagged",
            "scale": 80.0, "octaves": 8, "persistence": 0.65, "lacunarity": 2.5, "seed": 3000
        },
        {
            "label": "(d) High Frequency",
            "scale": 40.0, "octaves": 6, "persistence": 0.5, "lacunarity": 2.0, "seed": 4000
        }
    ]

    # Create the Plot (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    print("Generating maps...")
    
    for i, ax in enumerate(axes):
        p = samples[i]
        
        # Generate Map
        terrain = generate_perlin_map(p["scale"], p["octaves"], p["persistence"], p["lacunarity"], p["seed"])
        
        # Display Image
        ax.imshow(terrain, cmap='gray')
        
        # Remove axis ticks/numbers (clean look)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create Caption Text
        # We format it nicely for the article
        caption = (f"{p['label']}\n"
                   f"Scale: {p['scale']:.1f}\n"
                   f"Octaves: {p['octaves']}\n"
                   f"Persistence: {p['persistence']:.2f}\n"
                   f"Lacunarity: {p['lacunarity']:.2f}")
        
        # Add caption below the image
        ax.set_xlabel(caption, fontsize=12, labelpad=10)

    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    create_figure()