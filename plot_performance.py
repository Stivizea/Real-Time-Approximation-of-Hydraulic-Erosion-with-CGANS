import matplotlib.pyplot as plt
import numpy as np

# --- DATA FROM YOUR RUN ---
# Update these specific numbers if you have the exact seconds from your terminal
# Estimated based on 74x speedup:
# If AI took ~0.8 seconds -> Physics took ~59.2 seconds
ai_time = 0.8        
physics_time = ai_time * 74 
speedup = 74

def plot_comparison():
    # Setup
    labels = ['Physics Simulation\n(Sequential CPU)', 'cGAN Prediction\n(Parallel GPU)']
    times = [physics_time, ai_time]
    colors = ['#1f77b4', '#2ca02c'] # Blue for Physics, Green for AI

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create Bars
    bars = ax.bar(labels, times, color=colors, width=0.6)

    # Add Text Labels on top of bars
    # Physics Label
    ax.text(bars[0].get_x() + bars[0].get_width()/2., bars[0].get_height(),
            f'{physics_time:.2f} s',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # AI Label
    ax.text(bars[1].get_x() + bars[1].get_width()/2., bars[1].get_height(),
            f'{ai_time:.2f} s',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add Speedup Arrow
    # Coordinates for arrow
    x_start = 0 
    x_end = 1
    y_height = physics_time * 0.7 
    
    ax.annotate(f'{speedup}x Faster',
                xy=(x_end, y_height), xycoords='data',
                xytext=(x_start, y_height), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black', lw=1.5),
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkred')

    # Formatting
    ax.set_ylabel('Execution Time for 16 Maps (Seconds)', fontsize=12)
    ax.set_title('Performance Comparison: Physics vs. cGAN (Batch Size=16)', fontsize=16)
    
    # Use Log Scale because the difference is massive
    ax.set_yscale('log')
    
    # Tweak y-axis limits to look nice
    ax.set_ylim(0.1, physics_time * 2)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("performance_chart.png", dpi=300)
    print("Chart saved to performance_chart.png")
    plt.show()

if __name__ == "__main__":
    plot_comparison()