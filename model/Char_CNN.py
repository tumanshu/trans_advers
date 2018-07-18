import tensorflow as tf
from model.base_model import *


class CharCNN(object):
    """
    A CNN for text classification.
    based on the Very Deep Convolutional Networks for Natural Language Processing.
    """

    def __init__(self, conf, domain_adv, sentiment_dict, B_dict):
        all_dict = sentiment_dict+B_dict
        self.training = tf.placeholder(tf.int32, name="trainable_se")
        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True
        norm = tf.contrib.layers.variance_scaling_initializer()
        const = tf.constant_initializer(0.0)

        dict_len = tf.constant(len(sentiment_dict)+len(B_dict))


        self.input_x_se = tf.placeholder(tf.int32, [None, conf.sentence_size*conf.document_size], name="input_x_se")
        self.input_x_set = tf.placeholder(tf.int32, [None, conf.sentence_size*conf.document_size], name="input_x_set")

        self.se_y = tf.placeholder(tf.float32, [None, conf.sentiment_class], name="se_y")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob_se")
        self.l2_loss = tf.constant(0.0)
        self.encoding = position_encoding(conf.k_max_word, conf.embedding_size)



        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.embedded_characters = tf.nn.embedding_lookup(domain_adv.W0, self.input_x_se)
            self.embedded_characters_set = tf.nn.embedding_lookup(domain_adv.W_s_t, self.input_x_set)
            #self.embedded_characters = self.embedded_characters*domain_adv.encoding
            self.embedded_dict = tf.nn.embedding_lookup(domain_adv.W_s_t, sentiment_dict)
            self.embedded_dict_B = tf.nn.embedding_lookup(domain_adv.W_s_f, B_dict)
            self.embedded_dict = tf.concat([self.embedded_dict,self.embedded_dict_B],0)


            #self.embedded_dict = self.embedded_dict*self.encoding

            self.input_word_sentiment_mul = tf.einsum("ijk,lk->ijl", self.embedded_characters_set, self.embedded_dict)
            self.embedded_dict_sq = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_dict), axis=1))
            self.embedded_characters_sq = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_characters_set), axis=2))
            self.input_sq = tf.einsum("ij,l->ijl", self.embedded_characters_sq, self.embedded_dict_sq)
            self.input_word_sentiment = tf.truediv(self.input_word_sentiment_mul,self.input_sq)
            input_word_sentiment = tf.reshape(self.input_word_sentiment, [-1, conf.sentence_size*conf.document_size*(len(sentiment_dict)+len(B_dict))])


            self.max_value, self.max_indicate_ = tf.nn.top_k(input_word_sentiment, k=conf.k_max_word)
            self.max_indicate = tf.div(self.max_indicate_ - tf.mod(self.max_indicate_,dict_len),dict_len)
            self.max_indicaters = tf.mod(self.max_indicate_, dict_len)


            '''
            sentiment_dict_repeat = tf.tile(sentiment_dict, [conf.max_char_length_s])
            self.input_max = tf.gather(sentiment_dict_repeat, self.max_indicate)
            '''
            self.input_max = []
            self.input_max_id = []
            for i in range(conf.batch_size):
                input_max_batch = tf.gather(self.input_x_set[i], self.max_indicate[i])
                input_max_batch_id = tf.gather(tf.convert_to_tensor(all_dict),self.max_indicaters[i])
                self.input_max.append(input_max_batch)
                self.input_max_id.append(input_max_batch_id)

            self.input_max = tf.convert_to_tensor(self.input_max)
            self.input_max_id = tf.convert_to_tensor(self.input_max_id)

            '''
            input_max = tf.argmax(input_word_sentiment, axis=2)
            input_max = tf.gather(sentiment_dict,input_max)
            '''
            input_max_embedding = tf.expand_dims(tf.nn.embedding_lookup(domain_adv.W0, self.input_max)*self.encoding, -1)
            filter_dict = tf.get_variable("f_dict",  [1, conf.embedding_size, 1, conf.feature_map], initializer=norm)
            b_dict = tf.get_variable('b_dict', [conf.feature_map], initializer=const, trainable=TRAIN)

            self.feature_dict = Conv(input_max_embedding,filter_dict, [1,1,conf.embedding_size,1], TRAIN, 'layer_dict' )
            #self.feature_dict = tf.nn.relu(self.feature_dict+b_dict)
            self.feature_dict = tf.squeeze(tf.squeeze(average_pool(self.feature_dict, 'pooled_dict'), axis=1), axis=1)


            '''
            input_word_sentiment_max = tf.nn.softmax(tf.nn.max_pool(tf.transpose(tf.expand_dims(input_word_sentiment, 2)
                                        , perm=[0, 3, 2, 1]),ksize=[1, len(sentiment_dict),1,1],
                                        strides=[1, len(sentiment_dict),1, 1], padding='SAME'))
            


            self.feature_extern = tf.multiply(tf.squeeze(input_word_sentiment_max, axis=1),tf.transpose(
                                              self.embedded_characters, perm=[0,2,1]))

            '''
            #self.se_info, _ = linear(input_word_sentiment, conf.feature_map, TRAIN, 'se_liner')

            #attention_weight, output = attention(self.embedded_characters, self.se_info, conf.feature_map,
            #                                     "attention_1", TRAIN)


            self.embedded_characters_expanded = tf.expand_dims(self.embedded_characters, -1, name="embedding_input")

        with tf.variable_scope("layer-0"):
            
            strides0 = [1, 1, self.embedded_characters_expanded.get_shape().as_list()[2], 1]
            b_h0 = tf.get_variable('b_h0', [conf.feature_map], initializer=const, trainable=TRAIN)
            self.h0 = Conv(self.embedded_characters_expanded, domain_adv.filter_0, strides0, TRAIN, 'layer_0')
            self.h0 = tf.nn.relu(self.h0+b_h0)
        with tf.variable_scope("layer_1-29"):
            #b_h1 = tf.get_variable('b_h1', [conf.conf.feature_map], initializer=const, trainable=TRAIN)
            self.h1, _, _ = Convolutional_Block(self.h0, conf.feature_map, domain_adv.filter_1, domain_adv.filter_2,TRAIN, 'layer_1-2')


            self.pooled1 = tf.squeeze(tf.squeeze(average_pool(self.h1, 'pooled1'), axis=1), axis=1)
            #self.pooled1 = tf.concat([self.pooled1, self.se_info], axis=1)
            self.h2 = tf.concat([self.pooled1,self.feature_dict], 1)#
            print(self.h2.get_shape())
            '''
            #self.embedded_characters = tf.tanh(tf.einsum('hij,jk->hik', self.embedded_characters, domain_adv.Ws) + domain_adv.bs)
            self.hi = tf.tanh(tf.einsum('hij,jk->hik',self.embedded_characters,domain_adv.Ws)+domain_adv.bs)
            #batch_normal1 = tf.layers.batch_normalization(self.hi, trainable=TRAIN)
            #self.hi = tf.nn.relu(batch_normal1)


            self.a1, v_d_1 = attention(self.embedded_characters, self.hi, domain_adv.question)
            a1_liner = tf.matmul(domain_adv.question, domain_adv.wl) + domain_adv.bl
            self.hop2_input = v_d_1 + a1_liner

            #batch_normal2 = tf.layers.batch_normalization(self.hop2_input, trainable=TRAIN)
            #self.hop2_input = tf.nn.relu(batch_normal2)
            a2, v_d_2 = attention(self.embedded_characters, self.hi,self.hop2_input)
            a2_liner = tf.matmul(self.hop2_input, domain_adv.wl) + domain_adv.bl
            hop3_input = a2_liner + v_d_2

            #batch_normal3 = tf.layers.batch_normalization(hop3_input, trainable=TRAIN)
            #hop3_input = tf.nn.relu(batch_normal3)
            a3, v_d_3 = attention(self.embedded_characters, self.hi, hop3_input)
            a3_liner = tf.matmul(hop3_input, domain_adv.wl) + domain_adv.bl
            self.v_s_output = a3_liner + v_d_3
            '''

            '''
            pooled_1 = tf.nn.max_pool(self.h1, ksize=[1, conf.filter_size, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                    name="pool1")

            self.h2, _, _ = Convolutional_Block(pooled_1, 128, domain_adv.filter_3, domain_adv.filter_4, TRAIN, 'layer_3-4')
            self.pooled1 = tf.squeeze(tf.squeeze(average_pool(self.h2, 'pooled1'), axis=1), axis=1)
            
            pooled_2 = tf.nn.max_pool(self.h2, ksize=[1, conf.filter_size, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                                    name="pool2")

            self.h3, _, _ = Convolutional_Block(pooled_2, 256, None, TRAIN, 'layer_5-6')
            pooled_3 = tf.nn.max_pool(self.h3, ksize=[1, conf.filter_size, 1, 1], strides=[1, 2, 1, 1],
                                           padding='SAME', name="pool3")

            self.h4, _, _ = Convolutional_Block(pooled_3, 512, None, TRAIN, 'layer_7-8')

            #self.h5 = tf.transpose(self.h4, [0, 3, 2, 1])
            pooled4 = average_pool(self.h4, 'pooled4')
            self.h6 = tf.squeeze(tf.squeeze(pooled4, axis=1), axis=1)
            #self.h6 = tf.concat([tf.squeeze(tf.squeeze(pooled4, axis=1), axis=1), domain_adv.v_d_output], 1)
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
