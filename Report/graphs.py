import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Example data
# 14 values from 10000 to 290000
iterations = np.arange(10000, 290001, 10000)

# append the last one (297600)
iterations = np.append(iterations, 297600)
# PQ values for three different methods (replace with your data)
x = [
    21,
    24.703,
    27.099,
    29.802,
    32.218,
    33.954,
    35.093,
    35.969,
    37.093,
    38.446,
    39.013,
    40.059,
    40.379,
    41.056,
    40.983,
    42.294,
    42.856,
    43.448,
    44.755,
    44.382,
    45.032,
    45.442,
    44.685,
    45.507,
    45.308,
    46.244,
    46.295,
    46.995,
    46.776,
    47.085
]

y = [
    27.592,
    33.733,
    37.147,
    39.815,
    43.020,
    44.141,
    47.438,
    48.528,
    52.355,
    51.566,
    52.528,
    53.524,
    55.573,
    55.288,
    56.119,
    55.293,
    56.983,
    56.976,
    57.630,
    58.470,
    58.398,
    58.395,
    58.688,
    58.229,
    58.999,
    59.007,
    59.337,
    59.307,
    59.360,
    59.795
]


# Plot each method
plt.figure(figsize=(8,5))
plt.plot(iterations, x, marker='o', label="$10^{-5}$")
plt.plot(iterations, y, marker='o', label="3535 x $10^{-8}$")

# Labels and title
plt.xlabel("Επαναλήψεις")
plt.ylabel("PQ")
plt.title("PQ vs Επαναλήψεις για Διαφορετικές Τιμές του Ρυθμού Μάθησης")

# Legend and grid
plt.legend(title="Μέθοδος")
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()
