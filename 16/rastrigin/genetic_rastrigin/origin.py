#https://af-e.net/python-genetic-algorithms/#index_id14
import random
import numpy as np
def generate_initial_population(population_size, gene_length,bound):
    return [[np.random.uniform(-bound, bound) for _ in range(gene_length)] for _ in range(population_size)]
def rastrigin(x):
    A = 10
    x = np.asarray(x)  # 入力をNumPy配列に変換
    n = x.size         # 次元数
    term1 = A * n
    term2 = np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return term1 + term2
def fitness_function(individual):
    values = []

    tmp1 = rastrigin(individual)
    print(tmp1)
    return tmp1
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    return random.choices(population, weights=probabilities, k=2)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual
def next_generation(population, mutation_rate):
    new_population = []
    fitness_values = [fitness_function(ind) for ind in population]
    while len(new_population) < len(population):
        parent1, parent2 = selection(population, fitness_values)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutation(child1, mutation_rate))
        new_population.append(mutation(child2, mutation_rate))
    return new_population[:len(population)]
def termination_condition(generation, max_generations):
    return generation >= max_generations
# 終了条件の確認
if termination_condition(100, 200):
    print("終了条件を満たしました。")
# メイン処理
population_size = 10
gene_length = 6 #dim
max_generations = 1000
mutation_rate = 0.1
bound = 5.12
population = generate_initial_population(population_size, gene_length,bound)
for generation in range(max_generations):
    fitness_values = [fitness_function(ind) for ind in population]
    population = next_generation(population, mutation_rate)
    if termination_condition(generation, max_generations):
        print("終了条件を満たしました。")
        break
print("最終集団:", population)