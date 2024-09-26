import GPy
import GPyOpt
import numpy as np

trial = 1

# -2から2まで0.1刻みの数値を含む配列を作成
pre_x = np.arange(-7, -4, 0.2)
pre_x = pre_x.reshape(-1,1)

# 関数の計算
pre_y = np.cos(pre_x) * pre_x**-1 * np.sin(pre_x)  # 例としてsin関数を使用
pre_y = pre_y.reshape(-1,1)

# 目的関数の定義
def objective_function(x):
    
    global pre_y, pre_x, trial
    
    print(f'trial : {trial}')
    trial += 1
    
    #候補のxを出力して学習データに追加する
    print('x :',x)
    pre_x = np.append(pre_x,x)
    pre_x = pre_x.reshape(-1,1)
    
    #候補のxからyを計算して学習データに追加する
    y = np.cos(x) * x**-1 * np.sin(x)
    print('y :',y)
    pre_y = np.append(pre_y,y)
    pre_y = pre_y.reshape(-1,1)
    
    return y

# 探索領域の設定
domain = [{'name': 'x', 'type': 'continuous', 'domain': (-10, 10)}]

# BayesianOptimizationオブジェクトの作成
bo = GPyOpt.methods.BayesianOptimization(f=objective_function,
                                         model_type='GP',
                                         domain=domain,
                                        acquisition='EI',
                                        maximize = True,
                                        X=pre_x,
                                        Y=pre_y)
#　ベイズ最適化の実行（yが0.99以上になるまで繰り返す）
while np.max(pre_y)<0.99:

    # 1つの候補点を出力
    bo.run_optimization(max_iter=1)
    bo.plot_acquisition() #ガウス過程回帰のプロットと獲得関数のグラフを表示

#出力結果
#trial : 1
#x : [[10.]]
#y : [[0.04564726]]
#trial : 2
#x : [[6.12281422]]
#y : [[-0.02574559]]
#trial : 3
#x : [[-10.]]
#y : [[0.04564726]]
#trial : 4
#x : [[2.01567796]]
#y : [[-0.19271995]]
#trial : 5
#x : [[8.35341097]]
#y : [[-0.05032997]]
#trial : 6
#x : [[-9.04815013]]
#y : [[-0.03779874]]
#trial : 7
#x : [[-0.83117423]]
#y : [[0.59903928]]
#trial : 8
#x : [[-0.49667406]]
#y : [[0.84346902]]
#trial : 9
#x : [[0.06836471]]
#y : [[0.99688709]]