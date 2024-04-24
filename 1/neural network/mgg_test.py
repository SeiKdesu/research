import numpy as np

from nnga import NNGA
from Individual import Population
import crossover as cross
from mgg import MGG

# 交叉方法指定
crossType = cross.CrossoverType.UNDX
#致死個体処置方法
deadlyType = MGG.DeadlyIndividualType.Remove

#GAパラメータ
MAX_GENERATION  = 1000   #最大世代数
CROSS_TIMES     = 50    #交差回数
POPULATION_SIZE = 50    #個体サイズ

#MGG パラメータ
#エリート選択数
MGG_ELITE_SELECT_SIZE = 1

#ChoromIndex
CROME_SIZE = 1     #染色体(Chromosome) の 個数
VALUE_SIZE = 1     #評価値の数
cIndex = 0


if __name__ == "__main__":
    # NN
    nnga = NNGA()
    PARAMETER_SIZE = nnga.get_weights_size()

    # 評価関数に従った設定を取得
    dRange   = NNGA.get_range()
    evalType = NNGA.get_eval_type()

    mgg = MGG(dRange[0],dRange[1],dRange[0],dRange[1], crossType , 
              CROSS_TIMES , MGG_ELITE_SELECT_SIZE , deadlyType)

    #集団
    population = Population(POPULATION_SIZE , PARAMETER_SIZE , CROME_SIZE , VALUE_SIZE)

    # 交差方法初期化
    mgg.init_cross([PARAMETER_SIZE])

    #集団の初期化
    mgg.init_population(population)

    #初期集団 評価
    for ind in population.fIndividual:
        ind.fValue = nnga.eval_NNGA(ind.fChrom[cIndex])

    # 実行
    for gen  in  range(MAX_GENERATION+1):
        #MGGと交叉の実行
        mgg.make_children(population) 

        # 作成した子の評価
        for ind in mgg.fChildren.fIndividual:
            ind.fValue = nnga.eval_NNGA(ind.fChrom[cIndex])

        #MGGと交叉の実行
        mgg.select_children(population , evalType )

    #最終集団 評価
    for ind in population.fIndividual:
        ind.fValue = nnga.eval_NNGA(ind.fChrom[cIndex])
    #最終結果シミュレート
    bestIndex    = population.get_best_individual(evalType )
    nnga.finally_simulate (population.fIndividual[bestIndex].fChrom[cIndex] , 'sim' , True)
