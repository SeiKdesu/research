
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random
from icecream import ic
from smt.problems import Rosenbrock
from rbf_surrogate_100_train import predict_surrogate

# def Rosenbrock(x, n):
#     value = 0
#     for i in range(n-1):
#         value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
#     return value

# def rastrigin(x):
#     A = 10
#     x = np.asarray(x)  # 入力をNumPy配列に変換
#     n = x.size         # 次元数
#     term1 = A * n
#     term2 = np.sum(x**2 - A * np.cos(2 * np.pi * x))
#     return term1 + term2
# def objective_function_ga(x,dim):
#     n_rosenbrock = dim
   
#     rosen_value = rastrigin(x)
#     print(rosen_value)
#     return rosen_value
def Rosenbrock(x,n):
    value = 0
    for i in n:
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def Rosenbrock1(x):
    value = 0
    for i in range(3,4):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return value
def dixon_price(x,n):
    # n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in n])
    return term1 + term2

def objective_function_ga(x,dim):
    values=[]
    dim1 = [0,1]
    dim2 = [3,4]

    tmp1 = Rosenbrock(x,dim1)
    tmp2 = Rosenbrock(x,dim2)
    values.append(tmp1+tmp2)
    return np.array(values)

def objective_function_fit(x,dim):
    values=[]

    dim1= np.array((0,1,2,3,4))
    tmp1 = Rosenbrock(x,dim1)

    values.append(tmp1)
    return np.array(values)
# シード値の設定
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# パラメータの設定
dim = 6

max_gen = 100
pop_size = 100
offspring_size = 100
bound_rastrigin = 5.12
bound = 5.12 # Typical bound for Rosenbrock function

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    # return [objective_function_fit(individual,dim) for individual in population]
    return objective_function_ga(population,dim)


    # eva = np.abs(predict_surrogate(population))
    # return eva

# ルーレット選択
def roulette_wheel_selection(population, fitness):
    # current_best_fitness_index = np.argmin(fitness)
    # return population[current_best_fitness_index]
    max_val = sum(fitness)
    pick = random.uniform(0, max_val)
    current = 0
 

    fitness_roulette = np.array(fitness)
    fitness_roulette= np.squeeze(fitness_roulette)
    for i, f in enumerate(fitness_roulette):
        current += f
        
        if current > pick.any():
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

    #population=population[0]
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    best_fitness_history = []
    avg_fitness_history = []    

            
    for generation in range(max_gen):
        population = np.array(population)

        fitness = evaluate_population(population)
        ic(fitness)
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
        # population = sorted(population, key=lambda x: predict_surrogate(population))[:pop_size]
        fitness = np.squeeze(fitness)
        current_best_fitness = min(fitness)

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            index = np.argmin(fitness)
            best_individual = population[index]
            ic(best_fitness)
            pop_surrogate = np.zeros((100,6))
            pop_surrogate[0] = best_individual
            tmp_fitness = predict_surrogate(pop_surrogate)
            ic(tmp_fitness[0])

        # if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
        #     break

    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

# 実行
best_individual, best_fitness, best_fitness_history, avg_fitness_history = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)
print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")
# print(f"surrogate{predict_surrogate(best_individual)}")
print("objective function",objective_function_ga(best_individual,dim))
pop_surrogate = np.zeros((100,6))
pop_surrogate[0] = best_individual
tmp_fitness = predict_surrogate(pop_surrogate)
print('surrogate',tmp_fitness[0])
# print("suroogate",predict_surrogate(best_individual))
import matplotlib.pyplot as plt

def plot_fitness_history(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    # plt.ylim(0,10000)
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('not_opt_rosenbrock.pdf')
plot_fitness_history(best_fitness_history, avg_fitness_history)