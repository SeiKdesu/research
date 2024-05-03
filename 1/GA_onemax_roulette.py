import numpy as np
import matplotlib.pyplot as plt


class Individual:
    '''各個体のクラス
        args: 個体の持つ遺伝子情報(np.array)'''
    def __init__(self, genom):
        self.genom = genom
        self.fitness = 0  # 個体の適応度(set_fitness関数で設定)
        self.set_fitness()

    def set_fitness(self):
        '''個体に対する目的関数(OneMax)の値をself.fitnessに代入'''
        self.fitness = self.genom.sum()

    def get_fitness(self):
        '''self.fitnessを出力'''
        return self.fitness

    def mutate(self):
        '''遺伝子の突然変異'''
        tmp = self.genom.copy()
        #インデックスをどこか一つ指定する。ランダムに
        i = np.random.randint(0, len(self.genom) - 1)
        #ランダムで選んだインデックスの数列を１か所0と1を反転させる。
        tmp[i] = float(not self.genom[i])
        #突然変異によって変更させた個体を自分自身に書き込む
        self.genom = tmp
        #突然変異後の適合率を算出
        self.set_fitness()


def select_roulette(generation):
    '''選択の関数(ルーレット方式)'''
    selected = []
    weights = [ind.get_fitness() for ind in generation]
    norm_weights = [ind.get_fitness() / sum(weights) for ind in generation]
    selected = np.random.choice(generation, size=len(generation), p=norm_weights)
    return selected


def select_tournament(generation):
    '''選択の関数(トーナメント方式)'''
    selected = []
    for i in range(len(generation)):
        tournament = np.random.choice(generation, 3, replace=False)
        max_genom = max(tournament, key=Individual.get_fitness).genom.copy()
        selected.append(Individual(max_genom))
    return selected


def crossover(selected):
    '''交叉の関数'''
    children = []
    if POPURATIONS % 2:
        selected.append(selected[0])
    for child1, child2 in zip(selected[::2], selected[1::2]):
        if np.random.rand() < CROSSOVER_PB:
            child1, child2 = cross_two_point_copy(child1, child2)
        children.append(child1)
        children.append(child2)
    children = children[:POPURATIONS]
    return children


def cross_two_point_copy(child1, child2):
    '''二点交叉'''
    size = len(child1.genom)
    tmp1 = child1.genom.copy()
    tmp2 = child2.genom.copy()
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    new_child1 = Individual(tmp1)
    new_child2 = Individual(tmp2)
    return new_child1, new_child2


def mutate(children):
    #突然変異させる関数で一定の確率で突然変異させる。
    for child in children:
        #ランダムな値が一定の確率値よりも低ければ、突然変異関数を実行し、突然変異させる。
        if np.random.rand() < MUTATION_PB:
            child.mutate()
    return children


def create_generation(POPURATIONS, GENOMS):
    '''初期世代の作成
        return: 個体クラスのリスト'''
    generation = []
    for i in range(POPURATIONS):
        individual = Individual(np.random.randint(0, 2, GENOMS))
        generation.append(individual)
    return generation


def ga_solve(generation):
    '''遺伝的アルゴリズムのソルバー
        return: 最終世代の最高適応値の個体、最低適応値の個体'''
    best = []
    worst = []
    # --- Generation loop
    print('Generation loop start.')
    for i in range(GENERATIONS):
        # --- Step1. Print fitness in the generation
        best_ind = max(generation, key=Individual.get_fitness)
        best.append(best_ind.fitness)
        worst_ind = min(generation, key=Individual.get_fitness)
        worst.append(worst_ind.fitness)
        print("Generation: " + str(i) \
                + ": Best fitness: " + str(best_ind.fitness) \
                + ". Worst fitness: " + str(worst_ind.fitness))

        # --- Step2. Selection (Roulette)
        selected = select_roulette(generation)
        #selected = select_tournament(generation)

        # --- Step3. Crossover (two_point_copy)
        children = crossover(selected)

        # --- Step4. Mutation　突然変異
        generation = mutate(children)

    print("Generation loop ended. The best individual: ")
    print(best_ind.genom)
    return best, worst


np.random.seed(seed=65)

# param
POPURATIONS = 100 #１世代の個体数
GENOMS = 100 #遺伝子の長さ(n)
GENERATIONS = 20 #繰り返す世代数
CROSSOVER_PB = 0.8 #交叉を行う確率
MUTATION_PB = 0.1 #突然変異させる確率(ランダム値よりも低ければ突然変異)

# create first genetarion
generation = create_generation(POPURATIONS, GENOMS)

# solve  #適合率などを算出
best, worst = ga_solve(generation)

# plot
fig, ax = plt.subplots()
ax.plot(best, label='max')
ax.plot(worst, label='min')
ax.axhline(y=GENOMS, color='black', linestyle=':', label='true')
ax.set_xlim([0, GENERATIONS - 1])
ax.set_ylim([0, GENOMS * 1.1])
ax.legend(loc='best')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Tournament Select')
plt.savefig('GA_onemax_tournament_100')
plt.show()