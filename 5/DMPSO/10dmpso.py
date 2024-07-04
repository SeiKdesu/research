import numpy as np
import matplotlib.pyplot as plt

def original_rastrigin(x, n):
    value = 0
    for i in range(n):
        value += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    value += 10 * n
    return value

class Particle:
    def __init__(self, dimensions):
        self.position = np.random.uniform(-5.12, 5.12, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')
        self.lbest_position = None

    def update_velocity(self, w, c1, c2, r1, r2):
        inertia = w * self.velocity
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (self.lbest_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

class DMSPSO:
    def __init__(self, num_particles, dimensions, num_groups, max_iter, R):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.num_groups = num_groups
        self.max_iter = max_iter
        self.R = R
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.bounds = (-5.12, 5.12)
        self.particles = [Particle(dimensions) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.best_values = []

    def optimize(self):
        for iter in range(self.max_iter):
            # Group assignment
            groups = self.assign_groups()

            # Update lbest for each group
            for group in groups:
                lbest_position = None
                lbest_value = float('inf')
                for particle in group:
                    fitness = original_rastrigin(particle.position, self.dimensions)
                    if fitness < lbest_value:
                        lbest_value = fitness
                        lbest_position = particle.position.copy()
                for particle in group:
                    particle.lbest_position = lbest_position

            for particle in self.particles:
                fitness = original_rastrigin(particle.position, self.dimensions)
                if fitness < particle.pbest_value:
                    particle.pbest_value = fitness
                    particle.pbest_position = particle.position.copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                r1, r2 = np.random.random(), np.random.random()
                particle.update_velocity(self.w, self.c1, self.c2, r1, r2)
                particle.update_position(self.bounds)

            if iter % self.R == 0 and iter != 0:
                self.reassign_groups()

            self.best_values.append(self.global_best_value)

            if iter % 10 == 0:
                print(f"Iteration {iter+1}/{self.max_iter}, Best Value: {self.global_best_value}")

    def assign_groups(self):
        groups = [[] for _ in range(self.num_groups)]
        np.random.shuffle(self.particles)
        for i, particle in enumerate(self.particles):
            groups[i % self.num_groups].append(particle)
        return groups

    def reassign_groups(self):
        np.random.shuffle(self.particles)

# Example usage
num_particles = 80
dimensions = 30
num_groups = 4
max_iter = 500
R = 100

pso = DMSPSO(num_particles=num_particles, dimensions=dimensions, num_groups=num_groups, max_iter=max_iter, R=R)
pso.optimize()

# Plotting the best values
plt.plot(pso.best_values)
plt.xlabel('Iteration')
plt.ylabel('Best Value')
plt.title('DMS-PSO Optimization on Rastrigin Function')

plt.savefig("dmpso.pdf")
