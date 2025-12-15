import random
import math
import matplotlib.pyplot as plt

# Function to calculate distance between two cities
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Function to calculate the total distance of a path through cities
def total_distance(cities, cityorder):
    total = 0
    for i in range(len(cityorder) - 1):
        total += distance(cities[cityorder[i]], cities[cityorder[i + 1]])
    total += distance(cities[cityorder[-1]], cities[cityorder[0]])
    return total

def tsp(cities):
    N = len(cities)
    
    # Function to generate an initial solution (sequential order of cities)
    def initial_solution(N):
        return list(range(N))
    
    # Function to generate a random initial solution (shuffled order of cities)
    def random_initial_solution(N):
        initial_solution = list(range(N))
        random.shuffle(initial_solution)
        return initial_solution
    
    # Function to calculate the percentage improvement between two distances
    def percentage_improvement(random_total_dist, best_total_dist):
        return ((random_total_dist - best_total_dist) / random_total_dist) * 100

    # Initializing the current and best solutions
    current_order = initial_solution(N)
    current_distance = total_distance(cities, current_order)
    best_order = current_order
    best_distance = current_distance

    # Simulated Annealing parameters
    temperature = 1000
    cooling_rate = 0.999
    max_iterations = 10000

    for iteration in range(max_iterations):
        new_order = current_order.copy()

        i, j = sorted(random.sample(range(N), 2))
        
        # Reversing the order of cities between the selected indices
        new_order[i:j+1] = reversed(new_order[i:j+1])

        new_distance = total_distance(cities, new_order)

        delta_distance = new_distance - current_distance

        # Accepting the new order with a certain probability based on temperature
        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_order = new_order
            current_distance = new_distance

            if current_distance < best_distance:
                best_order = current_order
                best_distance = current_distance

        temperature *= cooling_rate
        
    # Generating a random initial order and calculate the total distance
    random_initial_order = random_initial_solution(N)
    random_total_dist = total_distance(cities, random_initial_order)

    return best_order, best_distance, random_total_dist, percentage_improvement(random_total_dist, best_distance)

# Reading city data from a file or provide a list of city coordinates
with open('tsp40.txt', 'r') as file:
    lines = file.readlines()
    num_cities = int(lines[0])
    cities = [tuple(map(float, line.strip().split())) for line in lines[1:]]

# Finding the order of visiting cities using simulated annealing
cityorder, best_total_dist, random_total_dist, improvement_percentage = tsp(cities)

print(f"Order of visiting cities: {cityorder}")
print(f"Total distance (Simulated Annealing): {best_total_dist}")
print(f"Total distance (Random Initial Order): {random_total_dist}")
print(f"Percentage Improvement: {improvement_percentage:.2f}%")

# Plotting the cities and the path
x, y = zip(*[cities[i] for i in cityorder])
x = list(x) + [x[0]]  # Close the loop
y = list(y) + [y[0]]  # Close the loop

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o')
plt.title(f'TSP Path | Total Distance: {best_total_dist:.4f}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
plt.savefig("figure")
