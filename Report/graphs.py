import matplotlib.pyplot as plt
import numpy as np

# Example data
# 14 values from 10000 to 140000
iterations = np.arange(10000, 140001, 10000)

# append the last one (148800)
iterations = np.append(iterations, 148800)
# PQ values for three different methods (replace with your data)
x = [
    29.215, 34.636, 38.009, 40.211, 43.68, 45.705,
    46.752, 49.964, 50.461, 52.189, 53.035,
    53.473, 53.686, 54.69, 55.365
]

y = [
    17.495, 21.852, 24.983, 28.493, 30.031, 31.725,
    33.57, 36.308, 38.035, 38.881, 39.043,
    40.559, 40.719, 41.802, 42.495
]

z  = [
    15.946, 21.153, 23.849, 27.126, 28.461, 29.909,
    31.5, 31.553, 34.98, 35.932, 35.798,
    37.162, 37.507, 38.563, 38.686
]


# Plot each method
plt.figure(figsize=(8,5))
plt.plot(iterations, x, marker='o', label="MLP4-BN")
plt.plot(iterations, y, marker='o', label="MLP4-GN")
plt.plot(iterations, z, marker='o', label="MLP4-BGN")


# Labels and title
plt.xlabel("Επαναλήψεις")
plt.ylabel("PQ")
plt.title("PQ vs Επαναλήψεις του MLP4 για Διαφορετικές Μεθόδους Κανονικοποίησης")

# Legend and grid
plt.legend(title="Μέθοδος")
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()
