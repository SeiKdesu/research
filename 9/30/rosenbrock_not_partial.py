
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random

from smt.problems import Rosenbrock


def Rosenbrock(x, n):
    value = 0
    for i in range(n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value


def objective_function(x,dim):
    n_rosenbrock = dim
   
    rosen_value = Rosenbrock(x, n_rosenbrock)
 
    return rosen_value

# パラメータの設定
dim = 10

max_gen = 900
pop_size = 200
offspring_size = 30
bound_rastrigin = 5.12
bound = 30  # Typical bound for Rosenbrock function

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual,dim) for individual in population]

# ルーレット選択
def roulette_wheel_selection(population, fitness):
    # current_best_fitness_index = np.argmin(fitness)
    # return population[current_best_fitness_index]
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
    alpha = 0.5 #親の情報をどれだけ持ってくるか
    beta = 0.35 #乱数をどれだけ受け入れるか
    g = 0.5 * (parent1 + parent2)
    d = parent2 - parent1
    norm_d = np.linalg.norm(d)
    if norm_d == 0:#parent2とparent1が等しいとき
        return parent1, parent2
    d = d / norm_d#どれだけ解に近いか。
    
    rand = np.random.normal(0, 1, dim)
    

    child1 = g + alpha * (parent3 - g) + beta * np.dot(rand, d) * d #乱数
    child2 = g + alpha * (g - parent3) + beta * np.dot(rand, d) * d
    for i in range(3,10):
        child1[i] = rand[i]
        child2[i] = rand[i]
    
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
    print(population)
    #population=population[0]
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    best_fitness_history = []
    avg_fitness_history = []    

            
    for generation in range(max_gen):
        
                
        fitness = evaluate_population(population)
        current_best_fitness = min(fitness)
        avg_fitness = np.mean(fitness)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)
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
        population = sorted(population, key=lambda x: objective_function(x,dim))[:pop_size]

        current_best_fitness = min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness.index(current_best_fitness)]
       
        # if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
        #     break

    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

# 実行
best_individual, best_fitness, best_fitness_history, avg_fitness_history = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)
print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")
print(objective_function(best_individual,dim))
import matplotlib.pyplot as plt

def plot_fitness_history(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.ylim(0,10000)
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('not_opt_rosenbrock.pdf')
plot_fitness_history(best_fitness_history, avg_fitness_history)