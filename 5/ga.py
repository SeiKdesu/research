import numpy as np
class Individual:
    def __init__(self, genom):
        self.genom = genom
        self.fitness = 0
        self.set_fitness()
    def set_fitness(self):
        self.fitness = self.genom.sum()

    def get_fitness(self):
        return self.fitness
    
def create_generation(POPURATIONS,GENOMS):
    generation = []
    for i in range(POPURATIONS):
        individual = Individual(np.random.randint(0,2,GENOMS))
        generation.append(individual)
    return generation

np.random.seed(seed=65)

POPURATIONS  =100
GENOMS= 50

generations= create_generation(POPURATIONS,GENOMS)
print(generations)

