# ✅ NumPy arrays & operations
# ✅ Matrix multiplication (foundation for neurons)
# ✅ Broadcasting (adding biases)
# ✅ Basic statistics (mean, std, variance)
# ✅ Visualizing data distributions

import numpy as np
import matplotlib.pyplot as plt

# Generate 100 values of x from 0 to 10
x = np.linspace(0, 10, 100)

# Generate random noise
noise = np.random.randn(100) * 2  # mean=0, std=2

# Simulate target values: y = 3x + 7 + noise
y = 3 * x + 7 + noise

# Plot
plt.scatter(x, y, color='teal', alpha=0.6)
plt.title("Simulated Linear Data with Noise")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.show()
