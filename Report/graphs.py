import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Example data
# 14 values from 10000 to 140000
iterations = np.arange(10000, 140001, 10000)

# append the last one (148800)
iterations = np.append(iterations, 148800)
# PQ values for three different methods (replace with your data)
x = [
    27.592, 33.733, 37.147, 39.815, 43.020,
    44.141, 47.438, 48.528, 52.355, 51.566,
    52.528, 53.524, 55.573, 55.288, 55.436
]

y = [
    26.641, 33.573, 37.123, 39.929, 43.482,
    46.132, 47.324, 47.998, 50.388, 52.798,
    53.952, 51.16, 53.751, 56.172, 55.123
]


# Plot each method
plt.figure(figsize=(8,5))
plt.plot(iterations, x, marker='o', label="VANmb-20000")
plt.plot(iterations, y, marker='o', label="VANmb-40000")

# Labels and title
plt.xlabel("Επαναλήψεις")
plt.ylabel("PQ")
plt.title("PQ vs Επαναλήψεις του MLP4 για Διαφορετικές Τιμές Επαναλήψεων Προσαρμογής της Θερμοκρασίας")

# Legend and grid
plt.legend(title="Μέθοδος")
plt.grid(True, linestyle="--", alpha=0.7)

plt.show()
