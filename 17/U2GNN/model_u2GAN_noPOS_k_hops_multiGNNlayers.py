import tensorflow as tf
from tensor2tensor.models import universal_transformer
#from tensor2tensor.models import transformer

epsilon = 1e-9

class u2GAN(object):
    def __init__(self, vocab_size, feature_dim_size, hparams_batch_size, ff_hidden_size, initialization, num_sampled,
                 seq_length, num_hidden_layers, k_num_GNN_layers=1):
        # Placeholders for input, output
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, seq_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, 1], name="input_y")
        # self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("input_feature"):
            self.input_feature = tf.compat.v1.get_variable(name="input_feature_1", initializer=initialization, trainable=False)

        #Inputs for Universal Transformer
        self.input_UT = tf.nn.embedding_lookup(self.input_feature, self.input_x)
        self.input_UT = tf.nn.l2_normalize(self.input_UT, axis=2)
        self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])

        self.hparams = universal_transformer.universal_transformer_small()
        self.hparams.hidden_size = feature_dim_size
        self.hparams.batch_size = hparams_batch_size * seq_length
        self.hparams.max_length = seq_length
        self.hparams.num_hidden_layers = num_hidden_layers # the number of self-attention layers, not the number of the GNN layers
        self.hparams.num_heads = 1 #due to the fact that the feature embedding sizes are various
        self.hparams.filter_size = ff_hidden_size
        self.hparams.use_target_space_embedding = False
        self.hparams.pos = None
        self.hparams.add_position_timing_signal = False
        self.hparams.add_step_timing_signal = False
        self.hparams.add_sru = False
        self.hparams.add_or_concat_timing_signal = None

        #
        for layer in range(k_num_GNN_layers):  # the number k of GNN layers, each GNN layer includes a number of self-attention layers
            # Universal Transformer Encoder
            self.ute = universal_transformer.UniversalTransformerEncoder(self.hparams, mode=tf.estimator.ModeKeys.TRAIN)
            self.output_UT = self.ute({"inputs": self.input_UT, "targets": 0, "target_space_id": 0})[0]
            self.output_UT = tf.squeeze(self.output_UT, axis=2)
            #
            self.output_target_node = tf.split(self.output_UT, num_or_size_splits=seq_length, axis=1)[0]
            self.output_target_node = tf.squeeze(self.output_target_node, axis=1)
            #input for next GNN hidden layer
            self.input_UT = tf.nn.embedding_lookup(self.output_target_node, self.input_x)
            self.input_UT = tf.reshape(self.input_UT, [-1, seq_length, 1, feature_dim_size])

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.compat.v1.get_variable(
                    "W", shape=[vocab_size, feature_dim_size], initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases, inputs=self.output_target_node,
                                       labels=self.input_y, num_sampled=num_sampled, num_classes=vocab_size))

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')

