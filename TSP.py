import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances

# Distance calculation
def calculate_tour_distance(tour, dist_matrix):
    indices = np.array(tour) - 1
    dist = dist_matrix[indices, np.roll(indices, -1)]
    return dist.sum()

# Tour perturbation
def generate_candidate(tour):
    n = len(tour)
    i, j = sorted(np.random.choice(range(n), 2, replace=False))
    candidate = tour.copy()
    candidate[i:j] = np.flip(candidate[i:j])
    return candidate

# Simulated annealing
def simulated_annealing(dist_matrix, max_iters=10000, T0=100, eta=0.99):

    n = len(dist_matrix)
    current = np.random.permutation(n) + 1

    best = current.copy()
    f_curr = calculate_tour_distance(current, dist_matrix)
    T = T0

    for i in range(max_iters):
        candidate = generate_candidate(current)
        f_cand = calculate_tour_distance(candidate, dist_matrix)

        delta_f = f_cand - f_curr
        if delta_f < 0 or np.random.rand() < math.exp(-delta_f / T):
            current = candidate
            f_curr = f_cand

        if f_cand < calculate_tour_distance(best, dist_matrix):
            best = candidate

        T *= eta

    return best

# (d) Generate random 50 city instance and run simulated annealing

num_cities = 50
cities = np.random.rand(num_cities, 2)
dist_matrix = pairwise_distances(cities)

print(f"Distance matrix for {num_cities} randomly generated cities:")
print(dist_matrix)

best_tour = simulated_annealing(dist_matrix)

print(f"\nBest tour found by simulated annealing:")
print(best_tour)

best_distance = calculate_tour_distance(best_tour, dist_matrix)
print(f"\nDistance of the best tour: {best_distance}")

# Visualization
plt.plot(cities[:, 0], cities[:, 1], 'o')
for i in range(len(best_tour)):
    plt.plot([cities[best_tour[i] - 1, 0], cities[best_tour[(i + 1) % num_cities] - 1, 0]],
             [cities[best_tour[i] - 1, 1], cities[best_tour[(i + 1) % num_cities] - 1, 1]], 'r-')

plt.show()
