import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances

# Distance calculation
def calculate_tour_distance(tour, dist_matrix):
    indices = np.array(tour) - 1
    dist = dist_matrix[indices, np.roll(indices, -1)]
    return dist.sum()

# Tour perturbation (original rule: reversing a subsequence)
def generate_candidate_reverse(tour):
    n = len(tour)
    i, j = sorted(np.random.choice(range(n), 2, replace=False))
    candidate = tour.copy()
    candidate[i:j] = np.flip(candidate[i:j])
    return candidate

# Tour perturbation (alternative rule: swapping two indices)
def generate_candidate_swap(tour):
    n = len(tour)
    i, j = np.random.choice(range(n), 2, replace=False)
    candidate = tour.copy()
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate

# Simulated annealing
def simulated_annealing(dist_matrix, max_iters=10000, T0=100, eta=0.99, proposal_rule=generate_candidate_reverse):
    n = len(dist_matrix)
    current = np.random.permutation(n) + 1
    best = current.copy()
    f_curr = calculate_tour_distance(current, dist_matrix)
    T = T0
    for i in range(max_iters):
        candidate = proposal_rule(current)
        f_cand = calculate_tour_distance(candidate, dist_matrix)
        delta_f = f_cand - f_curr
        if delta_f < 0 or np.random.rand() < math.exp(-delta_f / T):
            current = candidate
            f_curr = f_cand
        if f_cand < calculate_tour_distance(best, dist_matrix):
            best = candidate
        T *= eta
    return best

# Randomly generate city data
num_cities = 50
cities = np.random.rand(num_cities, 2)
dist_matrix = pairwise_distances(cities)

print(f"Distance matrix for {num_cities} randomly generated cities:")
print(dist_matrix)

# Original proposal rule: reversing a subsequence
best_tour_reverse = simulated_annealing(dist_matrix, proposal_rule=generate_candidate_reverse)
best_distance_reverse = calculate_tour_distance(best_tour_reverse, dist_matrix)

# Alternative proposal rule: swapping two indices
best_tour_swap = simulated_annealing(dist_matrix, proposal_rule=generate_candidate_swap)
best_distance_swap = calculate_tour_distance(best_tour_swap, dist_matrix)

# Print the comparison outcome
print(f"\nDistance of the best tour (original rule - reversing a subsequence): {best_distance_reverse}")
print(f"Distance of the best tour (alternative rule - swapping two indices): {best_distance_swap}")

# Visualization of the best tour from the original rule
plt.plot(cities[:, 0], cities[:, 1], 'o')
for i in range(len(best_tour_reverse)):
    plt.plot([cities[best_tour_reverse[i] - 1, 0], cities[best_tour_reverse[(i + 1) % num_cities] - 1, 0]],
             [cities[best_tour_reverse[i] - 1, 1], cities[best_tour_reverse[(i + 1) % num_cities] - 1, 1]], 'r-')

# Visualization of the best tour from the alternative rule
plt.plot(cities[:, 0], cities[:, 1], 'o')
for i in range(len(best_tour_swap)):
    plt.plot([cities[best_tour_swap[i] - 1, 0], cities[best_tour_swap[(i + 1) % num_cities] - 1, 0]],
             [cities[best_tour_swap[i] - 1, 1], cities[best_tour_swap[(i + 1) % num_cities] - 1, 1]], 'g-')

plt.show()
