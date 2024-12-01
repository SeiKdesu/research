import tensorflow as tf
import UT_NoPOS
import math

epsilon = 1e-9

class u2GAN(object):
    def __init__(self, num_hidden_layers, vocab_size, feature_dim_size, hparams_batch_size,
                 ff_hidden_size, initialization, num_sampled, seq_length):
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

        self.hparams = UT_NoPOS.universal_transformer_small1()
        self.hparams.hidden_size = feature_dim_size
        self.hparams.batch_size = hparams_batch_size * seq_length
        self.hparams.max_length = seq_length
        self.hparams.num_hidden_layers = num_hidden_layers
        self.hparams.num_heads = 1 #due to the fact that the feature embedding sizes are various
        self.hparams.filter_size = ff_hidden_size
        self.hparams.use_target_space_embedding = False

        #Universal Transformer Encoder
        self.ute = UT_NoPOS.UniversalTransformerEncoder1(self.hparams, mode=tf.estimator.ModeKeys.TRAIN)
        self.output_UT = self.ute({"inputs": self.input_UT, "targets": 0, "target_space_id": 0})[0]
        self.output_UT = tf.squeeze(self.output_UT)

        self.output_target_node = tf.split(self.output_UT, num_or_size_splits=seq_length, axis=1)[0]
        self.output_target_node = tf.squeeze(self.output_target_node)

        # self.output_target_node_batch = tf.tile(self.output_target_node, [1, seq_length-1])
        # self.output_target_node_batch = tf.reshape(self.output_target_node_batch, [-1, feature_dim_size])
        # self.output_target_node_batch = tf.nn.dropout(self.output_target_node_batch, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.compat.v1.get_variable(
                    "W", shape=[vocab_size, feature_dim_size], initializer=tf.contrib.layers.xavier_initializer())

            self.softmax_biases = tf.Variable(tf.zeros([vocab_size]))

        self.total_loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.embedding_matrix, biases=self.softmax_biases, inputs=self.output_target_node,
                                       labels=self.input_y, num_sampled=num_sampled, num_classes=vocab_size))

        self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.logging.info('Seting up the main structure')

