import numpy as np

# シグモイド 関数
def sigmoid(x):
    sigmoid_range = 34.538776394910684
    #オーバーフロー対策
    if x.any() <= -sigmoid_range:
        return 1e-15
    elif x.any() >= sigmoid_range:
        return 1.0 - 1e-15
    else:
        return 1.0 / (1.0 + np.exp(-x) )


# ソフトマックス 関数
def softmax(x):
    xmax = np.max(x) 
    exp_x = np.exp(x - xmax) 
    y = exp_x / np.sum(exp_x)
    return y

# Neural Network
class NN():

    #int    layer       : 層の数
    #int    input_size  : 入力層のニューロン数
    #int[]  hidden_size : 隠れ層のニューロン数
    #int    output_size : 出力層のニューロン数
    def __init__(self ,layer , input_size , hidden_size , output_size ):
        assert (layer >= 2 and input_size > 0 and output_size > 0) , 'ERROR:NN.init()'
        assert (layer-2 == len(hidden_size)) , 'ERROR:NN.init() hidden_size size'

        self.layer = layer
        self.input_unit  = input_size
        self.hidden_unit = hidden_size
        self.output_unit = output_size

        self.params = {}
        self.params['W1'] = np.zeros((input_size , hidden_size[0]) , dtype = 'float32')
        self.params['b1'] = np.zeros( hidden_size[0] , dtype = 'float32')
        self.params['W2'] = np.zeros((hidden_size[0] ,output_size), dtype = 'float32')
        self.params['b2'] = np.zeros( output_size , dtype = 'float32')


    #重み、バイアスの合計
    def get_weights_size(self) :
        w = 0
        for l in range(self.layer - 1):
            w_name = 'W' + str(l + 1)
            b_name = 'b' + str(l + 1)
            w += self.params[w_name].size
            w += self.params[b_name].size

        return w

    #重みの セット
    #numpy.ndarray weights : 重みとバイアス
    def set_weights(self , weights):
        w = 0

        for l in range(self.layer - 1):
            w_name = 'W' + str(l + 1)
            b_name = 'b' + str(l + 1)
            weight = weights[w: w+self.params[w_name].size]
            self.params[w_name] = weight.reshape(self.params[w_name].shape)
            w += self.params[w_name].size

            bias = weights[w: w+self.params[b_name].size]
            self.params[b_name] = bias.reshape(self.params[b_name].shape)
            w += self.params[b_name].size

        assert(weights.size == w) , \
            'Error:set_weights() weight size is invalid  weights = %d  nn = %d' %(weights.size , w)

    # 推論 

    def predict(self , x):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        i1 = np.dot(x,W1) + b1
        o1 = sigmoid(i1)
        i2 = np.dot(o1,W2) + b2
        y  = softmax(i2)

        return y
