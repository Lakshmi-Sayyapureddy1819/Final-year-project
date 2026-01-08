import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate dummy juvenile density heatmap values
data = np.random.rand(10, 10)

try:
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, cmap="YlGnBu")
    plt.title("Juvenile Density Heatmap Example")
    plt.show(block=False)
    input("Press ENTER to close heatmap window...")
except KeyboardInterrupt:
    plt.close()
