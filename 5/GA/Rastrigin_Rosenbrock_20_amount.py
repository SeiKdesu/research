import numpy as np
import random
import matplotlib.pyplot as plt

# Rastrigin関数の定義
def Rastrigin(x, n):
    value = 0
    for i in range(n):
        value += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    value += 10 * n
    return value

# Rosenbrock関数の定義
def Rosenbrock(x, n):
    value = 0
    for i in range(n - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

# オブジェクト関数の定義
def objective_function(x):
    n_rastrigin = 10
    n_rosenbrock = 10
    rastrigin_value = Rastrigin(x[:n_rastrigin], n_rastrigin)
    rosenbrock_value = Rosenbrock(x[n_rastrigin:], n_rosenbrock)
    return rastrigin_value + rosenbrock_value

# パラメータの設定
dim = 20
max_gen = 400
pop_size = 300
offspring_size = 200
bound_rastrigin = 5.12
bound_rosenbrock = 30  # Typical bound for Rosenbrock function

# 初期集団の生成
def init_population(pop_size, dim, bound_rastrigin, bound_rosenbrock):
    population = []
    for _ in range(pop_size):
        individual = np.zeros(dim)
        individual[:10] = np.random.uniform(-bound_rastrigin, bound_rastrigin, 10)
        individual[10:] = np.random.uniform(-bound_rosenbrock, bound_rosenbrock, 10)
        population.append(individual)
    return population

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual) for individual in population]

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
def mutate(individual, bound_rastrigin, bound_rosenbrock, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if i < 10:
                individual[i] = random.uniform(-bound_rastrigin, bound_rastrigin)
            else:
                individual[i] = random.uniform(-bound_rosenbrock, bound_rosenbrock)
    return individual

# メインの遺伝的アルゴリズム
def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound_rastrigin, bound_rosenbrock):
    population = init_population(pop_size, dim, bound_rastrigin, bound_rosenbrock)
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    avg_fitness_history = []

    for generation in range(max_gen):
        fitness = evaluate_population(population)

        if generation % 10 == 0:
            avg_fitness = np.mean(fitness)
            print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

        fitness_history.append(min(fitness))
        avg_fitness_history.append(np.mean(fitness))

        new_population = []
        while len(new_population) < offspring_size:
            parent1 = roulette_wheel_selection(population, fitness)
            parent2 = roulette_wheel_selection(population, fitness)
            parent3 = roulette_wheel_selection(population, fitness)
            child1, child2 = undx_crossover(parent1, parent2, parent3, dim)
            new_population.append(mutate(child1, bound_rastrigin, bound_rosenbrock))
            if len(new_population) < offspring_size:
                new_population.append(mutate(child2, bound_rastrigin, bound_rosenbrock))

        population = population + new_population
        population = sorted(population, key=lambda x: objective_function(x))[:pop_size]

        current_best_fitness = min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness.index(current_best_fitness)]

        if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
            break

    return best_individual, best_fitness, fitness_history, avg_fitness_history

# 実行
best_individual, best_fitness, fitness_history, avg_fitness_history = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound_rastrigin, bound_rosenbrock)
print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")

# グラフの作成
plt.figure(figsize=(12, 6))
plt.plot(fitness_history, label='Best Fitness')
plt.plot(avg_fitness_history, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')

plt.ylim(0,250)
plt.title('Composite_20')
plt.legend()
plt.grid(True)
plt.savefig('amount20_2000.pdf')
