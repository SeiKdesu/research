import gym

import numpy as np

import random

import torch #pytrochのimport
import torch.nn as nn #pytrochのライブラリの基本的なニューラルネットワーク

from tqdm import tqdm #進捗状況や処理状況を可視化するもの。

import typing
from typing import List

import matplotlib.pyplot as plt

Parameters = List[torch.nn.parameter.Parameter] #neural networkのパラメータを格納しておく

if torch.cuda.is_available():#gpuが使えるかどうかを確認する
    device = "cpu"
else:
    device = "cpu"

print(f"Pytorch will use device {device}")#デバイス名を出力

# turn off gradients as we will not be needing them
torch.set_grad_enabled(False) #自動微分を無効にする。推論のコードは自分で記述する。

env = gym.make('CartPole-v0') #opengymの中でも今回使うゲームはカートポールで行う。
obs = env.reset() #環境をすべてリセットする。

in_dim = len(obs) #入力の次元数をin_dimに格納する。
out_dim = env.action_space.n #環境からうける可能な行動数。（出力は右と左の2つになる）

print(in_dim, out_dim) #入力と出力の値を計算する

def get_params(net: torch.nn.Sequential) -> Parameters: #変数netがpytrochのsequetial型であることを示している。
    '''
    Gets the parameters from a PyTorch model stored as an nn.Sequential #Sequentialに格納されているパラメータをℓ・

    @params
        network (nn.Sequential): A pytorch model #引数はpytrochのモデルであり、
    @returns
        Parameters: the parameters of the model #戻り値はモデルのパラメータ
    '''
    params = [] #パラメータを格納する配列を用意する。

    for layer in net: #net(Sequential分だけ繰り返す)
        if hasattr(layer, 'weight') and layer.weight != None: #hasattr関数は指定されたオブジェクトが属性を持っているか調べることで、layerというオブジェクトにweightというメソッドがあるかを調べる
            params.append(layer.weight)#layerの重みをparamsにappendする
        if hasattr(layer, 'bias') and layer.bias != None:#hasattr関数でバイアスがあるか調べ、バイアスをparamsにappendする
            params.append(layer.bias)

    return params  #pytroch.nnの重みとバイアスが含まれた配列paramsが戻り値


def set_params(net: torch.nn.Sequential, params: Parameters) -> torch.nn.Sequential:
    '''
    Sets the parameters for an nn.Sequential　パラメータをモデルにセットする

    @params
        network (torch.nn.Sequential): A network to change the parameters of #パラメータが変わったpytrochのネットワーク
        params (Parameters): Parameters to place into the model   モデルに埋め込むパラメータの配列
    @returns
        torch.nn.Sequential: A model the the provided parameters パラメータを埋め込んでネットワークを返す
    '''
    i = 0 #iを0から始める
    for layerid, layer in enumerate(net): #layeridにはインデックスが付与され、layerにはnetのリストが代入され、ループされる
        if hasattr(layer, 'weight') and layer.weight != None: #layerに重みがあれば、paramsのi番目とnetのlayerid番目に代入
            net[layerid].weight = params[i]
            #print(torch.sign(net[layerid].weight))

            torch.sign(net[layerid].weight)
            #
            #print('wieghtです')
            #print(net[layerid].weight)
            #ここがBNN
            '''
            tensor_data=net[layerid].weight


            tensor_data = torch.where(tensor_data < 0, 0.0,1.0)

            net[layerid].weight = torch.nn.Parameter(tensor_data)
            '''
            i += 1 #代入したら、インデックスを上げる。

        if hasattr(layer, 'bias') and layer.bias != None:
            net[layerid].bias = params[i] #baisも同じように行う。

            i += 1
    return net #パラメータを埋め込んだネットワークを返す
#############################################################################################################################
import itertools


def quantize(value, min_val, max_val, num_bits):
    """
    Quantize a value to a given number of bits.

    Args:
        value (torch.Tensor): The value to quantize (can have multiple elements).
        min_val (float): The minimum value of the quantization range.
        max_val (float): The maximum value of the quantization range.
        num_bits (int): The number of bits to use for quantization.

    Returns:
        torch.Tensor: The quantized tensor with the same shape as the input.
    """

    # Calculate the quantization step size
    step_size = (max_val - min_val) / (2**num_bits - 1)

    # Quantize the value element-wise
    quantized_value = torch.div(value - min_val, step_size)

    # Clamp the quantized value to the valid range
    quantized_value = torch.clamp(quantized_value, min=0, max=2**num_bits - 1)

    return quantized_value.long()  # Convert to integer type (optional)
def quantize_tensor(tensor, min_val, max_val, num_bits):
    """
    Quantize a 2D tensor to a given number of bits.

    Args:
        tensor (torch.Tensor): The tensor to quantize.
        min_val (float): The minimum value of the quantization range.
        max_val (float): The maximum value of the quantization range.
        num_bits (int): The number of bits to use for quantization.

    Returns:
        torch.Tensor: The quantized tensor.
    """

    # Check if the input tensor is a 2D tensor
    if not isinstance(tensor, torch.Tensor) or len(tensor.shape) != 2:
        raise ValueError("Input tensor must be a 2D tensor.")

    # Flatten the tensor within the function
    flattened_tensor = tensor.flatten()

    # Quantize the flattened tensor element-wise
    quantized_values = quantize(flattened_tensor, min_val, max_val, num_bits)

    # Reshape the quantized values directly using PyTorch reshape
    quantized_tensor = quantized_values.reshape(tensor.shape)

    return quantized_tensor
def quantize_tensor_to_2bit(tensor, min_val, max_val):
  """
  Quantize a 2D tensor to 2 bits and convert it to a 2-bit binary representation.

  Args:
      tensor (torch.Tensor): The input tensor.
      min_val (float): The minimum value of the quantization range.
      max_val (float): The maximum value of the quantization range.

  Returns:
      str: The 2-bit binary representation of the quantized tensor.
  """

  # Quantize the tensor to 6 bits (intermediate step)
  num_bits = 6
  quantized_tensor = quantize_tensor(tensor, min_val, max_val, num_bits)

  # Convert quantized tensor to integer (element-wise)
  int_values = quantized_tensor.int()  # Element-wise conversion to integer
  i=0
  # Generate 2-bit binary strings (element-wise)
  binary_strings = []
  binary_representation = []
  for value in int_values:

    # Extract individual elements from the tensor using torch.flatten()
    count = int_values.shape[0]
    flat_values = value.flatten()
    for int_value in flat_values:

      # Convert 6-bit value to binary string (remove '0b' prefix)
      binary_string = bin(int_value)[2:]

      # Pad the binary string with leading zeros to 2 bits

      #ここから
      binary_string = binary_string.zfill(6)
       #binary_strings.append(binary_string)
      row_digits = [[int(c)] for c in binary_string]
      binary_representation.append(row_digits)


  return binary_representation

##################################################################################################################################################
def fitness(solution: Parameters, net: torch.nn.Sequential, render=True) -> float:
    '''
    Evaluate a solution, a set of weights and biases for the network　改善した（solution)の評価するネットワークの重みとバイアスを

    @params
        solution (Parameters): parameters to test the fitness of　テストデータによる適応度を算出したもの。
        net (torch.nn.Sequential): A network for testing the parameters with
        render (bool): whether or not to draw the agent interacting with the environment as it trains
    @returns
        float: The fitness of the solution　適応度が戻り値
    '''
    net = set_params(net, solution) #ネットワークと改善したparamesが引数になり、新たなnetworkを設定する。

    ob = env.reset() #環境をすべてリセットする

    done = False
    sum_reward = 0 #報酬の合計を格納する変数
    i=0 #####################################################
    #ob = torch.tensor(ob).float().unsqueeze(0)
    while not done: #while Trueのようなもの。
      
        #ob = torch.tensor(ob).float().unsqueeze(0).to(device)
        # obがリストのリストまたはNumPyのndarrayのリストの場合
        if i == 0:

            # popで取得した観測値のリスト全体を単一のNumPyのndarrayに変換する
            ob = np.array(ob[0])

            # NumPyのndarrayをPyTorchのテンソルに変換し、次元を追加してバッチサイズを1に設定し、デバイスに送信する
            ob = torch.tensor(ob).unsqueeze(0)

        
        #tensor型は機会学習や数値計算などに使われる手法のこと
        
        min_val = -3.0  # Minimum value of the quantization range
        max_val = 3.0  # Maximum value of the quantization range
        # Quantize the tensor to 2 bits and get the binary representation
        binary_representation = quantize_tensor_to_2bit(ob, min_val, max_val)
        
        binary_representation = torch.tensor(binary_representation)
        
        binary_representation = binary_representation.reshape(1,-1)
        

        ob = binary_representation.float()
    
        ob = ob.to(device)
        
        i +=1 
        
        q_vals = net(ob) #tensorで配列の形状を整えた入力をnetに入れ、pytrochのnnのSequentialの結果が戻り値となる
        
        act = torch.argmax(q_vals.cpu()).item() #netに入れたもののうちq_valueが最も高い部分を行動として選択。
        tmp = np.array(env.step(act)[0])
        if tmp[1] < 0:
            tmp[1] = 0
        ob_next, reward, done, info = tmp #選択された行動から次の観測値を割り出す。
        ob = ob_next #得られた観測値を今の状態とする。
        i += 1
        sum_reward += reward #報酬を加算する。
        if render: #もし動きを描画するようだったらここで描画する
            env.render()
    return sum_reward #合計の報酬が戻り値となる。

def select(pop: List[Parameters], fitnesses: np.ndarray) -> List[Parameters]:  #新たな個体を選択する関数
    '''
    Select a new population

    @params
        pop (List[Parameters]): The entire population of parameters　個体全体のパラメータ
        fitnesses (np.ndarray): the fitnesses for each entity in the population #全体の適応度を算出
    @returns
        List[Parameters]: A new population made of fitter individuals 新たな適切な個体を作る。
    '''
   
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitnesses/fitnesses.sum()) #np.ra
    return [pop[i] for i in idx] #

def crossover(parent1: Parameters, pop: List[Parameters]) -> Parameters:
    '''
    Crossover two individuals and produce a child.交叉をさせ新たな子を生成する

    This is done by randomly splitting the weights and biases at each layer for the parents and then
    combining them to produce a child　ランダムに重みとバイアスをそれぞれのレイヤーで分け、子を生成するときに合わせる

    @params
        parent1 (Parameters): A parent that may potentially be crossed over　交叉を行う親の個体
        pop (List[Parameters]): The population of solutions　　　　　　　　　解候補のポピュレーション（親個体を含む）
    @returns
        Parameters: A child with attributes of both parents or the original parent1　交叉された子個体。
    '''
    if np.random.rand() < CROSS_RATE:  #ランダムな確率に基づいて交叉を行う。
        i = np.random.randint(0, POP_SIZE, size=1)[0] #np.random.randint(最小値,最大値,出力される配列のshape)のうち０番目を見付ける
        parent2 = pop[i]                              #ランダムに選択されたインデックスを第二のおやとする。
        child = [] #子を格納する場所を用意しておき
        split = np.random.rand()  #0から1までの一様分布を１つ生成する。

        for p1l, p2l in zip(parent1, parent2):  #複数のリストの要素を同時に取得する。親１と親２のリストを同時に取得する。
            splitpoint = int(len(p1l)*split)    #親１の数×splitをする戻り値をintで表す。
            new_param = nn.parameter.Parameter(torch.cat([p1l[:splitpoint], p2l[splitpoint:]])) #splitointのインデックスまでを親１とし、splitpoint以降を親2とする。torchcatは配列を結合しnnの新たなパラメータする
            child.append(new_param) #新たなパラメータとして子どもに結合すｒ

        return child #子どもを戻り値とする。
    else:
        return parent1 #ランダムな確率に入らなければそのまま親を戻り値とする


def gen_mutate(shape: torch.Size) -> torch.tensor:
    '''
    Generate a tensor to use for random mutation of a parameter　ランダムな確率に基づいて突然変異させる

    @params
        shape (torch.Size): The shape of the tensor to be created　入力はtensorのshapeのサイズ
    @returns
        torch.tensor: a random tensor　ランダムなtensorを戻り値とする
    '''
    return nn.Dropout(MUTATION_RATE)(torch.ones(shape)-0.5) * torch.randn(shape).to(device)*MUTATION_FACTOR #torch.randnは平均0分散1の正規分布から乱数が生成される。そこにグローバル変数MUTATION_FACTORを掛ける。
    #確率MUTATION_RATEの確率でtensorの一部分が0になる。
def mutate(child: Parameters) -> Parameters:
    '''
    Mutate a child　子の突然変異させる

    @params
        child (Parameters): The original parameters　子のパラメータ
    @returns
        Parameters: The mutated child　突然変異させた子
    '''
    for i in range(len(child)):
        for j in range(len(child[i])):
            child[i][j] += gen_mutate(child[i][j].shape)

    return child #突然変異させた子を返す

#%%time

# hyperparameters for genetic algorithm
POP_SIZE = 60  #個体の要素数
CROSS_RATE = 0.8 #交叉する割合
MUTATION_RATE = 0.03 #突然変異する割合
MUTATION_FACTOR = 0.5  #突然変異するときに関係するハイパパラメータ
N_GENERATIONS = 100 #世代数
FITNESS_EARLY_STOP_THRESH = 500  #適応度のストップこの値を超えるとループ文をbreakする

# the pytorch neural network to train
net = nn.Sequential(nn.Linear(24, 18, bias=True),  #全結合層（入力の次元数、出力の次元数、バイアスがtrueかflaseか）
                    nn.Tanh(),                         #活性化関数ReLuを使う
                    nn.Linear(18, out_dim, bias=True)).to(device) #全結合層32,中間層16,バイアスがtrueのもの

                    #nn.Tanh(), #活性化関数にReLU
                    #nn.Linear(18, out_dim, bias=True)).to(device)  #全結合層で入力16出力はoutputの次元数でバイアスをtrueに

# get the required parameter shapes
base = get_params(net)  #定義したネットワークのパラメータを取得。重みやバイアスがパラメータとなりbaseに格納される。
shapes = [param.shape for param in base]    #baseから各sequentialごとのパラメータのshapeをshapesに格納する。

# build a population
pop = []           #個体を格納すpopを用意する。
for i in range(POP_SIZE):     #popsize今回は100まで繰り返す。
    entity = []     #全体を格納するentityを用意する
    for shape in shapes:    #paramsのshapeが格納されている配列から１つずつ配列を取り出す。
        # if fan in and fan out can be calculated (tensor is 2d) then using kaiming uniform initialisation
        # as per nn.Linear
        # otherwise use uniform initialisation between -0.5 and 0.5
        #networkのパラメータの初期値について２つの手法で行っている。
        #try:
            #rand_tensor = nn.init.kaiming_uniform_(torch.empty(shape)).to(device)
        #except ValueError:
        rand_tensor = nn.init.uniform_(torch.empty(shape), -1.0, 1.0).to(device)
        entity.append((torch.nn.parameter.Parameter(rand_tensor)))  #networkのパラメータの初期値を格納する。
    pop.append(entity)#初期値を格納したentityをpopに格納する。

# whether or not to render while training (false runs code a lot faster)#コードを早く動かすためにrenderはFalseにする。
render = False

# the max episodes (200 is the environment default)
env._max_episode_steps = 500 #最大のエピソード数は200にする。

average=[]
best_fit=[]
num=[]

# train
for i in range(N_GENERATIONS):  #世代数だけ繰り返す
    # get fitnesses適応度を算出
    fitnesses = np.array([fitness(entity, net, render) for entity in pop]) 
    # calculate average fitness of population
    avg_fitness = fitnesses.sum()/len(fitnesses)  #平均の適応度を計算

    # print info of generation
    print(f"Generation {i}: Average Fitness is {avg_fitness} | Max Fitness is {fitnesses.max()}")  #世代ごとに平均の適応度、最大の適応度をprintする。
    average.append(avg_fitness)
    best_fit.append(fitnesses.max())
    num.append(i)
    if avg_fitness > FITNESS_EARLY_STOP_THRESH:  #もしEARYstopよりも平均適応度が大きくなっていれば、すべての世代数をみずにbreakする
        break
    # select a new population新たな個体を選択。

    fittest = pop[fitnesses.argmax()] #適応度が一番よいpopulationを選択。
    pop = select(pop, fitnesses)      
    random.shuffle(pop)               #popをランダムにシャッフルする。
    pop = pop[:-1]                    #シャッフルしたすべてのpopを取得する。
    pop.append(fittest)               #一番適応度がよかったものをpopに結合する。
    pop2 = list(pop)                  #配列をlistの形に変換してpop2とする。
    # go through the population and crossover and mutate交叉と突然変異を行う。
    for i in range(len(pop)):#popの個数分だけ繰り返す。
        child = crossover(pop[i], pop2)#select関数の中からランダムに選択されたものを使って、新たな個体を生成するアルゴリズム
        child = mutate(child)          #生成した子を突然変異させる。
        pop[i] = child                 #交叉や生成、突然変異などをさせたはずの（一定の確率ではしないため）子をpopに格納する。

    #print(average,i)
plt.plot(num,average,color='blue')
#plt.scatter(num,average)
plt.ylim([0,500])
plt.title('FItnessAvg-SimpleGA-NN-CartPole-v2-Roulette')
plt.xlabel('GEnerations')
plt.ylabel('fitnessaverage')
plt.savefig('GA_モデル2')
#plt.show()

plt.plot(num,best_fit,color='red')
#plt.scatter(num,average)
plt.ylim([0,520])
plt.title('FItnessMax-SimpleGA-NN-CartPole-v1-Roulette')
plt.xlabel('GEnerations')
plt.ylabel('fitness_best')
#plt.show()