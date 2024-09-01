
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random

from smt.problems import Rosenbrock
from bentch_mark import objective_function3
def Rosenbrock(x, n,dim):
    if dim==0:

        value = 0
        for i in [0,3,5,9,10,11,14,16,17,18,21,22,23,25,26,28,29,30,31,32,34,35,36,37,40,46,49]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    elif dim==1:
        
        value = 0
        for i in [24,43,47,48]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    elif dim==2:
        
        value = 0
        for i in [1,2,4,6,7,8,12,13,19,20,27,33,38,39,41,42,44,45]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

def dixon_price(x,n,dim):
    if dim==0:
      
        n = len(x)
        term1 = (x[0] - 1) ** 2
        term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in [58,66,68,71,81,82,86,97,98]])
    elif dim==1:
        
        n = len(x)
        term1 = (x[0] - 1) ** 2
        term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in [53,54,55,57,60,61,63,64,67,69,75,76,77,79,80,85,87,90,92,95,99]])
    elif dim==2:
        
        n = len(x)
        term1 = (x[0] - 1) ** 2
        term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in [50,52,56,59,62,65,70,72,73,74,78,83,84,88,89,91,93,94,96]])

    return term1 + term2
def Rosenbrock1(x, n,dim):
    if dim==0:

        value = 0
        for i in [110,111,114,115,121,123,125,126,130,144]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    elif dim==1:
        
        value = 0
        for i in [104,132,146]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    elif dim==2:
        
        value = 0
        for i in [100,101,102,103,105,106,107,108,109,112,113,116,117,118,119,120,122,124,127,128,129,131,133,134,135,136,137,138,139,140,141,142,143,145,147,148,149]:
            value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
# def booth(xy):
#     """
#     Booth関数を計算します。

#     引数:
#     xy : array-like
#         入力ベクトル [x, y]
    
#     戻り値:
#     float
#         Booth関数の値
#     """
#     x, y = xy[0], xy[1]
    
#     term1 = x + 2*y - 7
#     term2 = 2*x + y - 5
    
#     return term1**2 + term2**2

# def matyas_function(x):
#     return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


# def trid(x,dim):
#     """
#     指定された次元数Dに対してTrid関数を計算します。

#     引数:
#     n : int
#         次元数（変数の数）
    
#     戻り値:
#     float
#         Trid関数の値
#     """
#     i=dim
#     if i==0:

#         # 入力ベクトル x を生成（例として 1 から n までの整数）
    
        
#         # 最初の和の計算
#         sum1 = 0
#         for i in [67,69,71,72,73,74,75,76,85,87,89,90]:
#             sum1 += (x[i] - 1)**2
        
#         # 2つ目の和の計算（隣接する要素の積）
#         sum2 = 0
#         for i in [67,69,71,72,73,74,75,76,85,87,89,90]:
#             sum2 += x[i] * x[i-1]
#     elif i ==1:
#            # 入力ベクトル x を生成（例として 1 から n までの整数）

        
#         # 最初の和の計算
#         sum1 = 0
#         for i in [60,63,65,66,77,79,81,82,84,86,91]:
#             sum1 += (x[i] - 1)**2
        
#         # 2つ目の和の計算（隣接する要素の積）
#         sum2 = 0
#         for i in [60,63,65,66,77,79,81,82,84,86,91]:
#             sum2 += x[i] * x[i-1]
#     elif i==2:
#            # 入力ベクトル x を生成（例として 1 から n までの整数）
#         # x = [61,62,64,68,70,78,80,83,88]
        
#         # 最初の和の計算
#         sum1 = 0
#         for i in [61,62,64,68,70,78,80,83,88]:
#             sum1 += (x[i] - 1)**2
        
#         # 2つ目の和の計算（隣接する要素の積）
#         sum2 = 0
#         for i in [61,62,64,68,70,78,80,83,88]:
#             sum2 += x[i] * x[i-1]

#     return sum1 - sum2
def objective_function(x,num):
    n_rosenbrock = 50
    n_dixon=50
    n_powell=32
    rosen_value = Rosenbrock(x, n_rosenbrock,num)
    dixon_value = dixon_price(x,n_dixon,num)
    rosen_value2 = Rosenbrock1(x, n_rosenbrock,num)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + dixon_value+rosen_value2

# パラメータの設定
dim = 151

max_gen = 3000
pop_size = 5
offspring_size = 200
bound_rastrigin = 5.12
bound = 30  # Typical bound for Rosenbrock function
np.random.seed(5555)
# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population,num):
    return [objective_function(individual,num) for individual in population]

# ルーレット選択
def roulette_wheel_selection(population, fitness):
    # num_elites=1
    #    # フィットネスに基づいて個体をソート（昇順）
    # sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
    
    # # 最もフィットネス値が低い個体を選択する
    # elites = [population[i] for i in sorted_indices[:num_elites]]
    
    # return elites if num_elites > 1 else elites[0]
    current_best_fitness_index = np.argmin(fitness)
    return population[current_best_fitness_index]
    # max_val = sum(fitness)
    # pick = random.uniform(0, max_val)
    # current = 0
    # for i, f in enumerate(fitness):
    #     current += f
    #     if current > pick:
    #         return population[i]
    # return population[-1]

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

    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    best_fitness_history = []
    avg_fitness_history = []    
    tmp1=0
    tmp2=0
    tmp3=0
    for i in range(3):
        best_fitness = float('inf')
        for generation in range(max_gen):
            
                    
            fitness = evaluate_population(population,i)
            
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
            population = sorted(population, key=lambda x: objective_function(x,i))[:pop_size]

            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                current_best_fitness_index = np.argmin(fitness)
       
                best_individual = population[current_best_fitness_index]
                # print(population[fitness.index(current_best_fitness)])
            # if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
            #     break
        if i == 0:
            print(i,'です。')
            tmp1=best_individual.copy()
            print('i=0のとき',best_individual)
            print('i=0のとき',tmp1[0])
        elif i == 1:
            tmp2=best_individual.copy()
            print('i=1のとき',best_individual)
        elif i==2:
            tmp3=best_individual.copy()
            for i in [0,1,3,4,8,9,11,12,13,14,17,18,19,20,21,22,24,28,29,67,69,71,72,73,74,75,76,85,87,89,90]:
                best_individual[i]=tmp1[i]

            for i in [5,6,7,10,23,26,30,31,33,36,37,38,40,41,42,43,45,46,47,49,50,52,55,57,58,60,63,65,66,77,79,81,82,84,86,91] :
                best_individual[i]=tmp2[i]
            for i in [2,15,16,25,27,44,48,54,56,61,62,64,68,70,78,80,83,88]:
                best_individual[i]=tmp3[i]
            print('i=2のとき',best_individual)
        print(tmp1,tmp2,tmp3)
    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

# 実行
best_individual, best_fitness, best_fitness_history, avg_fitness_history = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)
print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")
print(objective_function3(best_individual,151))
import matplotlib.pyplot as plt

def plot_fitness_history(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('opt_rosenbrock_1000.pdf')
plot_fitness_history(best_fitness_history, avg_fitness_history)