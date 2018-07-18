import tensorflow as tf
from model.base_model import *
from tensorflow.contrib.rnn import DropoutWrapper, GRUCell
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn

def attention(inputs, size, scope):
    with tf.variable_scope(scope):
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[size],
                                                   regularizer=layers.l2_regularizer(scale=L2_REG),
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, size,
                                                  activation_fn=tf.tanh,
                                                  weights_regularizer=layers.l2_regularizer(scale=L2_REG))
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2,
                                    keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(inputs, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)

    return outputs


def bi_gru_encode(rnn_size, inputs, sentence_size, scope=None):
    # batch_size = inputs.get_shape()[0]

    with tf.variable_scope(scope or 'bi_gru_encode'):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

        enc_out, (enc_state_fw, enc_state_bw) = bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                          cell_bw=bw_cell,
                                                                          inputs=inputs, dtype=tf.float32,
                                                                          sequence_length=sentence_size)

        enc_state = tf.concat([enc_state_fw, enc_state_bw], 1)
        enc_outputs = tf.concat(enc_out, 2)

    return enc_outputs, enc_state


class HAN(object):
    """
    A CNN for text classification.
    based on the Very Deep Convolutional Networks for Natural Language Processing.
    """

    def __init__(self, conf, sentiment_dict, num_quantized_chars, word2vec):
        norm = tf.contrib.layers.variance_scaling_initializer()
        const = tf.constant_initializer(0.0)
        dict_len = tf.constant(len(sentiment_dict))
        self.input_x_se = tf.placeholder(tf.int32, [None, conf.sentence_size*conf.document_size], name="input_x_se")
        self.input_x_set = tf.placeholder(tf.int32, [None, conf.sentence_size*conf.document_size], name="input_x_set")
        self.sentence_len = tf.placeholder(tf.int32, [None, conf.document_size], name="sentence_len")
        self.doc_len = tf.placeholder(tf.int32, [None,], name="sentence_len")

        self.se_y = tf.placeholder(tf.float32, [None, conf.sentiment_class], name="se_y")
        self.training = tf.placeholder(tf.int32, name="trainable_ddv")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob_adv")
        self.l2_loss = tf.constant(0.0)
        self.encoding = position_encoding(conf.k_max_word, conf.embedding_size)

        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True

        if len(word2vec) == 0:
            self.W0 = tf.get_variable("W", [num_quantized_chars, conf.embedding_size], trainable=TRAIN)
        else:
            self.W0 = tf.get_variable("W", initializer=word2vec, trainable=TRAIN)
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.embedded_characters = tf.nn.embedding_lookup(self.W0, self.input_x_se)
            self.embedded_characters_set = tf.nn.embedding_lookup(self.W0, self.input_x_set)
            # self.embedded_characters = self.embedded_characters*domain_adv.encoding
            self.embedded_dict = tf.nn.embedding_lookup(self.W0, sentiment_dict)
            # self.embedded_dict = self.embedded_dict*self.encoding

            self.input_word_sentiment = tf.einsum("ijk,lk->ijl", self.embedded_characters_set, self.embedded_dict)
            self.embedded_dict_sq = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_dict), axis=1))
            self.embedded_characters_sq = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_characters_set), axis=2))
            self.input_sq = tf.einsum("ij,l->ijl", self.embedded_characters_sq, self.embedded_dict_sq)
            self.input_word_sentiment = self.input_word_sentiment / self.input_sq
            input_word_sentiment = tf.reshape(self.input_word_sentiment,
                                              [-1, conf.sentence_size*conf.document_size * len(sentiment_dict)])

            self.max_value, self.max_indicate = tf.nn.top_k(input_word_sentiment, k=conf.k_max_word)
            self.max_indicate = tf.div(self.max_indicate - tf.mod(self.max_indicate, dict_len), dict_len)
            '''
            sentiment_dict_repeat = tf.tile(sentiment_dict, [conf.max_char_length_s])
            self.input_max = tf.gather(sentiment_dict_repeat, self.max_indicate)
            '''
            self.input_max = []
            for i in range(conf.batch_size):
                input_max_batch = tf.gather(self.input_x_set[i], self.max_indicate[i])
                self.input_max.append(input_max_batch)

            self.input_max = tf.convert_to_tensor(self.input_max)

            '''
            input_max = tf.argmax(input_word_sentiment, axis=2)
            input_max = tf.gather(sentiment_dict,input_max)
            '''
            input_max_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.W0, self.input_max) * self.encoding,
                                                 -1)
            filter_dict = tf.get_variable("f_dict", [1, conf.embedding_size, 1, conf.feature_map], initializer=norm)
            b_dict = tf.get_variable('b_dict', [conf.feature_map], initializer=const, trainable=TRAIN)

            self.feature_dict = Conv(input_max_embedding, filter_dict, [1, 1, conf.embedding_size, 1], TRAIN,
                                     'layer_dict')
            # self.feature_dict = tf.nn.relu(self.feature_dict+b_dict)
            self.feature_dict = tf.squeeze(tf.squeeze(average_pool(self.feature_dict, 'pooled_dict'), axis=1), axis=1)




            with tf.variable_scope('character_encoder') as scope:
                char_length = tf.reshape(self.sentence_len, [-1])  # [batch_size * document_size]
                char_inputs = tf.reshape(self.embedded_characters, [-1, conf.sentence_size, conf.embedding_size])
                char_outputs, _ = bi_gru_encode(conf.rnn_size, char_inputs, char_length, scope)
                with tf.variable_scope('attention') as scope:
                    char_attn_outputs = attention(char_outputs, conf.word_attention_size, scope)
                    char_attn_outputs = tf.reshape(char_attn_outputs,
                                                   [-1, conf.document_size, char_attn_outputs.shape[-1]])

                with tf.variable_scope('dropout'):
                    char_attn_outputs = layers.dropout(char_attn_outputs,
                                                       keep_prob=self.keep_prob,
                                                       is_training=TRAIN)

            with tf.variable_scope('sentence_encoder') as scope:

                sent_outputs, _ = bi_gru_encode(conf.rnn_size, char_attn_outputs, self.doc_len, scope)

                with tf.variable_scope('attention') as scope:
                    sent_attn_outputs = attention(sent_outputs, conf.sent_attention_size, scope)

                with tf.variable_scope('dropout'):
                    self.sent_attn_outputs = layers.dropout(sent_attn_outputs,
                                                       keep_prob=self.keep_prob,
                                                       is_training=TRAIN)
                    self.sent_attn_outputs = tf.concat([self.sent_attn_outputs, self.feature_dict],1)
                    '''
                                    layers.fully_connected(inputs=sent_attn_outputs,
                                                                  num_outputs=self.num_class,
                                                                  activation_fn=None,
                                                                  weights_regularizer=layers.l2_regularizer(scale=L2_REG))
                    '''




    def se_loss(self, feature, num_class):
        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True

        if TRAIN:
            feature = tf.nn.dropout(feature, self.keep_prob)

        self.fc3_out, self.l2_loss = fc_layer(feature, num_class, self.l2_loss, "fc-1-2-3_se", TRAIN)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3_out, labels=self.se_y)

        self.predictions = tf.argmax(self.fc3_out, 1, name="predictions")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.se_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return tf.reduce_mean(losses), self.l2_loss
