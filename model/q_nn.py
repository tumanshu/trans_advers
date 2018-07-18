import tensorflow as tf
from model.base_model import *


class q_nn(object):
    def __init__(self, conf, num_quantized_chars, word2vec, word2vec_s):
        self.encoding = position_encoding(conf.sentence_size*conf.document_size, conf.embedding_size)
        const = tf.constant_initializer(0.0)
        self.input_x_d = tf.placeholder(tf.int32, [None, conf.sentence_size*conf.document_size], name="input_x_d")

        self.d_y = tf.placeholder(tf.float32, [None, conf.domain_class], name="domain_y")

        self.training = tf.placeholder(tf.int32, name="trainable")
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss = tf.constant(0.0)

        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True

        norm = tf.contrib.layers.variance_scaling_initializer()
        self.question = tf.get_variable("question", [1, conf.feature_map], trainable=TRAIN)
        if len(word2vec) == 0:
            self.W0 = tf.get_variable("W", [num_quantized_chars, conf.embedding_size], trainable=TRAIN)
            self.W_s = tf.get_variable("W_s", [num_quantized_chars, conf.embedding_size], trainable=False)
        else:
            self.W0 = tf.get_variable("W", initializer=word2vec, trainable=TRAIN)
            self.W_s_f = tf.get_variable("W_s_f", initializer=word2vec_s, trainable=False)
            self.W_s_t = tf.get_variable("W_s_t", initializer=word2vec_s, trainable=True)

        self.Ws = tf.get_variable("Ws",[conf.embedding_size, conf.feature_map], initializer=norm,trainable=TRAIN)
        self.bs = tf.get_variable("bs", [conf.feature_map], initializer=const, trainable=TRAIN)

        self.wl = tf.get_variable('wl', [conf.feature_map, conf.feature_map], initializer=norm, trainable=TRAIN)
        self.bl = tf.get_variable('bl', [conf.feature_map], initializer=const, trainable=TRAIN)


        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_characters = tf.nn.embedding_lookup(self.W0, self.input_x_d)
            self.embedded_characters = self.embedded_characters*self.encoding
            self.embedded_characters_expanded = tf.expand_dims(self.embedded_characters, -1, name="embedding_input")

        with tf.variable_scope('layer_0'):
            filter_shape0 = [conf.filter_size, conf.embedding_size, 1, conf.feature_map]
            strides0 = [1, 1, conf.embedding_size, 1]
            self.filter_0 = tf.get_variable('filter1', filter_shape0, initializer=norm)
            self.h0 = Conv(self.embedded_characters_expanded, self.filter_0, strides0, TRAIN, 'layer_0')
        with tf.variable_scope('layer_1-2'):
            self.h1, self.filter_1, self.filter_2 = Convolutional_Block(self.h0, conf.feature_map, None, None,TRAIN, 'layer_1-2')
            self.pooled1 = tf.squeeze(tf.squeeze(average_pool(self.h1, 'pooled1'), axis=1), axis=1)
            '''
            pooled_1 = tf.nn.max_pool(self.h1, ksize=[1, conf.filter_size, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                      name="pool1")

            self.h2, self.filter_3, self.filter_4 = Convolutional_Block(pooled_1, 128, None, None,TRAIN, 'layer_3-4')
            self.pooled1 = tf.squeeze(tf.squeeze(average_pool(self.h2, 'pooled1'), axis=1), axis=1)
            '''

        #self.question = rnn(self.question, conf.feature_map)
        '''
        with tf.name_scope('attention'):

            self.hi = tf.tanh(tf.einsum('hij,jk->hik',self.embedded_characters,self.Ws)+self.bs)
            #self.hi = tf.tanh(self.embedded_characters*self.Ws+self.bs)
            #batch_normal1 = tf.layers.batch_normalization(self.hi, trainable=TRAIN)
            #self.hi = tf.nn.relu(batch_normal1)


            self.a1, v_d_1 = attention(self.embedded_characters, self.hi, self.question)

            a1_liner = tf.matmul(self.question, self.wl) + self.bl
            self.hop2_input = a1_liner + v_d_1
            #batch_normal2 = tf.layers.batch_normalization(self.hop2_input, trainable=TRAIN)
            #self.hop2_input = tf.nn.relu(batch_normal2)


            a2, v_d_2 = attention(self.embedded_characters, self.hi, self.hop2_input)
            a2_liner= tf.matmul(self.hop2_input, self.wl) + self.bl
            hop3_input = a2_liner + v_d_2
            #batch_normal3 = tf.layers.batch_normalization(hop3_input, trainable=TRAIN)
            #hop3_input = tf.nn.relu(batch_normal3)


            a3, v_d_3 = attention(self.embedded_characters,self.hi, hop3_input)
            a3_liner = tf.matmul(hop3_input, self.wl) + self.bl
            self.v_d_output = a3_liner + v_d_3
        '''
    def adversarial_loss(self, feature,  num_class):

        feature = flip_gradient(feature)
        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True

        if TRAIN:
            feature = tf.nn.dropout(feature, self.keep_prob)

        # Map the features to TASK_NUM classes
        logits, loss_l2 = fc_layer(feature, num_class, self.l2_loss, "fc-1-2-3_adv", TRAIN)
        loss_adv = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.d_y, logits=logits))
        self.predictions = tf.argmax(logits, 1, name="predictions_adv")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.d_y, 1))
        self.accuracy_adv = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy_adv")

        return loss_adv, loss_l2


