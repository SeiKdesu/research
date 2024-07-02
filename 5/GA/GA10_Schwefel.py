import numpy as np
import random

# Schwefel関数の定義
def Schwefel(x, n):
    value = 0
    for i in range(n):
        value += x[i] * np.sin(np.sqrt(np.absolute(x[i])))
    value = 418.9828873 * n - value
    return value

# パラメータの設定
dim = 10
max_gen = 10000
pop_size = 300
offspring_size = 200
bound = 500

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [Schwefel(individual, dim) for individual in population]

# ルーレット選択
def roulette_wheel_selection(population, fitness):
    max_val = sum(fitness)
    pick = random.uniform(0, max_val)
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current > pick:
            return population[i]
    return population[-1]

# UNDX交叉操作
def undx_crossover(parent1, parent2, parent3, dim):
    alpha = 0.5
    beta = 0.35
    g = 0.5 * (parent1 + parent2)
    d = parent2 - parent1
    norm_d = np.linalg.norm(d)
    if norm_d == 0:
        return parent1, parent2
    d = d / norm_d
    rand = np.random.normal(0, 1, dim)
    child1 = g + alpha * (parent3 - g) + beta * np.dot(rand, d) * d
    child2 = g + alpha * (g - parent3) + beta * np.dot(rand, d) * d
    return child1, child2

# 変異操作
def mutate(individual, bound, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-bound, bound)
    return individual

# メインの遺伝的アルゴリズム
def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []

    for generation in range(max_gen):
        fitness = evaluate_population(population)

        if generation % 100 == 0:
            avg_fitness = np.mean(fitness)
            print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

        fitness_history.append(np.mean(fitness))

        new_population = []
        while len(new_population) < offspring_size:
            parent1 = roulette_wheel_selection(population, fitness)
            parent2 = roulette_wheel_selection(population, fitness)
            parent3 = roulette_wheel_selection(population, fitness)
            child1, child2 = undx_crossover(parent1, parent2, parent3, dim)
            new_population.append(mutate(child1, bound))
            if len(new_population) < offspring_size:
                new_population.append(mutate(child2, bound))

        population = population + new_population
        population = sorted(population, key=lambda x: Schwefel(x, dim))[:pop_size]

        current_best_fitness = min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness.index(current_best_fitness)]

        if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
            break

    return best_individual, best_fitness, fitness_history

# 実行
best_individual, best_fitness, fitness_history = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)
print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")
