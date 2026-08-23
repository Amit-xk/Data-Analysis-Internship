import matplotlib.pyplot as plt
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_path = os.path.join(project_path, "Figures")

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_text):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    for n, (layer_size, text) in enumerate(zip(layer_sizes, layer_text)):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
      
        ax.text(n*h_spacing + left, bottom - 0.05, text, ha='center', va='top', fontsize=10)
        
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='gray', alpha=0.3)
                ax.add_artist(line)

fig = plt.figure(figsize=(12, 8))
ax = fig.gca()
ax.axis('off')

layer_sizes = [7, 8, 6, 4, 1]
texts = [
    "Input Layer\n(14 Features)",
    "Hidden Layer 1\n(128 Units)\nBatchNorm + ReLU\nDropout (0.2)",
    "Hidden Layer 2\n(64 Units)\nBatchNorm + ReLU\nDropout (0.2)",
    "Hidden Layer 3\n(32 Units)\nBatchNorm + ReLU\nDropout (0.1)",
    "Output Layer\n(1 Unit)\nLinear"
]

draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, texts)
plt.title("Deep Feedforward Neural Network Architecture", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()

output_file = os.path.join(figures_path, "dl_architecture.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Success! Diagram saved to:\n{output_file}")