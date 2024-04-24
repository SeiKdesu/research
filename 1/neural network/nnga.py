import numpy as np
#OpenAIGym
import gym

from nn import NN

#重み の範囲
WEIGHT_MIN  = -5.0
WEIGHT_MAX  =  5.0

#評価値小 ＝ 優良個体
EVAL_TYPE_MIN = 0
#評価値大 ＝ 優良個体
EVAL_TYPE_MAX = 1

#層の数
LAYER_SIZE = 3
#入力層のニューロン数
INPUT_SIZE = 4
#出力層のニューロン数
OUTPUT_SIZE = 2

HIDDEN_SIZE = [10]

# 20step
SIMULATE_STEP = 200


#角度[rad]を-π～πに変換
def rad_conversion(rad):
    rad += np.pi;
    rad %= 2*np.pi;
    if rad < 0:
        rad += np.pi
    else:
        rad -= np.pi
    return rad


# Neural Network
class NNGA():
    #探索範囲取得
    def get_range() :
        ret = [WEIGHT_MIN , WEIGHT_MAX]
        return ret

    #評価タイプ取得
    def get_eval_type() :
        ret = EVAL_TYPE_MIN
        return ret

    def __init__(self ):
        self.nn = NN(LAYER_SIZE , INPUT_SIZE ,  HIDDEN_SIZE , OUTPUT_SIZE )

        print("Use Open AI Gym CartPole")
        self.env = gym.make('CartPole-v0')

    #重み、バイアスの合計
    def get_weights_size(self) :
        return self.nn.get_weights_size()

    #評価
    #weights : 重み
    def eval_NNGA(self ,weights ) :
        #重みの セット
        self.nn.set_weights(weights)

        #評価用シミュレート
        eval = self.eval_simulate()
        return eval

    #最終結果シミュレート
    #weights : 重み
    def finally_simulate(self , weights , filename, use_gym_render = False):
        #重みの セット
        self.nn.set_weights(weights)

        # シミュレート
        self.save_simulate( filename , use_gym_render)

    ##########################################################
    ### 倒立振子
    ##########################################################
    # 1stepシミュレート
    def simulate_1step(self , step ):
        inputs = np.reshape( self.state , (1 , INPUT_SIZE))

        #NNを使って推論 
        y = self.nn.predict(inputs)
        # 振り子アクション判定
        if(y[0][0] > y[0][1]):
            action = 0
        else:
            action = 1

        # 振り子シミュレート
        state , reward ,done ,dummy = self.env.step(action)
        state[2] =  rad_conversion(state[2])
        reward = -reward

        return state , reward

    #評価
    def eval_simulate(self ) :
        #振り子初期化
        self.state = self.env.reset()

        eval = 0
        for step in range(SIMULATE_STEP):
            self.state ,reward = self.simulate_1step(step )
            eval += reward

        return eval


    #シミュレート 動画作成
    def save_simulate(self ,  filename , use_gym_render) :
        #振り子初期化
        self.state = self.env.reset()

        eval = 0
        for step in range(SIMULATE_STEP):
            if (use_gym_render == True):
                self.env.render()

            self.state , reward=self.simulate_1step(step)
            eval += reward

            print('step={0}  state={1} reward={2}'.format(step,self.state,eval))

