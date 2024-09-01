import pygad
import numpy

# Rastrigin関数を定義
def rastrigin_function(solution):
    A = 10
    n = len(solution)
    return A * n + numpy.sum(solution**2 - A * numpy.cos(2 * numpy.pi * solution))

# フィットネス関数を定義（最小化を最大化に変更）
def fitness_func(ga_instance, solution, solution_idx):
    output = rastrigin_function(solution)
    # Rastriginの最小化をフィットネスの最大化に変更
    fitness = 1.0 / (output + 0.000001)
    return fitness

fitness_function = fitness_func

num_generations = 100  # 世代数
num_parents_mating = 7  # 交配する親の数

# 初期集団のサイズ
sol_per_pop = 50
# 遺伝子の数（変数の数）
num_genes = 6

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")
    last_fitness = ga_instance.best_solution()[1]

# GAクラスのインスタンスを作成
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       on_generation=callback_generation)

# 遺伝的アルゴリズムの実行
ga_instance.run()

# フィットネス値の進化をプロット
ga_instance.plot_fitness()

# 最良の解を取得
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# 最良の解で予測を実行
prediction = rastrigin_function(solution)
print(f"Predicted output based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# GAインスタンスを保存
filename = 'genetic_rastrigin'
ga_instance.save(filename=filename)

# 保存されたGAインスタンスをロード
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness()
