import numpy as np


import matplotlib.pyplot as plt


from smt.surrogate_models import IDW, RBF, RMTC, RMTB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smt.utils.misc import compute_rms_error

def Rosenbrock(x, n):
    value = 0
    for i in range(n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2

   
    return value
# def dixon_price(x,n):
#     n = len(x)
#     term1 = (x[0] - 1) ** 2
#     term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(11, 20)])
#     return term1 + term2
def Rosenbrock1(x, n):
    value = 0
    for i in range(3,n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

def objective_function(p):
    x=p
    n_rosenbrock = 3
    # n_dixon=10
    # n_powell=32
    rosen_value = Rosenbrock(x, n_rosenbrock)
    # dixon_value = dixon_price(x,n_dixon)
    rosen_value2 = Rosenbrock1(x, n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + rosen_value2

# %%
from sko.GA import GA
from sko.PSO import PSO
dim=6
def make_data():
    ga = GA(func=objective_function, n_dim=dim, size_pop=10, max_iter=10, prob_mut=0.001, lb=[-1]*dim, ub=[1]*dim, precision=1e-7)
    best_x, best_y = ga.run()



    print('best_x:', best_x, '\n', 'best_y:', best_y)

    # %% Plot the result


    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig('a.png')


    # pso = PSO(func=objective_function, dim=dim, pop=3, max_iter=150, lb=[-1]*dim, ub=[1]*dim, w=0.8, c1=0.5, c2=0.5)
    # pso.run()
    #  print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)


    def pop ():
        population = ga.X
        population= np.squeeze(population)
        return population
    def fitness():
        fitness = ga.Y
        return fitness

    population = pop()
    fitness= fitness()
    return population,fitness
population,fitness = make_data()

from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}'


from scikit import *

xt = np.array(population, dtype=np.double)    
yt = np.array(fitness, dtype=np.double)

np.savetxt(f"acc_loss/{name}_pop.txt", xt, fmt='%.6f')  # フォーマットを指定

t = RBF(print_prediction=False, poly_degree=0)
t.set_training_values(xt, yt)

t.train()
xtest = np.array(population, dtype=np.double)    
ytest = np.array(fitness, dtype=np.float32)
# Prediction of the validation points
y = t.predict_values(xtest)
print("RBF,  err: " + str(compute_rms_error(t, xtest, ytest)))


