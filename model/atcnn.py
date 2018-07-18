import tensorflow as tf
from model.base_model import *


class atCNN(object):
    def __init__(self, conf, num_quantized_chars):
        self.input_x_d = tf.placeholder(tf.int32, [None, conf.max_char_length_d], name="input_x_d")
        self.input_x_se = tf.placeholder(tf.int32, [None, conf.max_char_length_s], name="input_x_se")

        self.training = tf.placeholder(tf.int32, name="trainable")

        self.list_d_s = [self.input_x_d, self.input_x_se]
        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True


        self.l2_loss = tf.constant(0.0)
        self.W0 = tf.get_variable("W", [num_quantized_chars, conf.embedding_size],)
        self.all_conv_1 = []
        norm = tf.random_normal_initializer(stddev=0.1)
        for i in range(len(self.list_d_s)):
            with tf.variable_scope('%d' % (i)):
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    self.embedded_characters = tf.nn.embedding_lookup(self.W0, self.list_d_s[i])
                    self.embedded_characters_expanded = tf.expand_dims(self.embedded_characters, -1, name="embedding_input")

                with tf.variable_scope('layer_0'):
                    filter_shape0 = [conf.filter_size, conf.embedding_size, 1, 64]
                    strides0 = [1, 1, conf.embedding_size, 1]
                    self.filter_0 = tf.get_variable('filter1', filter_shape0, initializer=norm)
                    self.h0 = Conv(self.embedded_characters_expanded, self.filter_0, strides0, TRAIN, 'layer_0')
                    self.all_conv_1.append(self.h0)
                '''
                with tf.variable_scope('layer_1-2'):
                    self.h1 = Convolutional_Block(self.h0, 64, TRAIN, 'layer_1-2')
                    #self.pooled_1 = tf.nn.max_pool(self.h1, ksize=[1, conf.filter_size, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                    #                 name="pool1")
                    self.all_conv_1.append(self.h1)
                '''
        with tf.name_scope('att_layer_3-8'):

            #part of domain classificatin
            #attention_1
            A = distance(self.all_conv_1[0], self.all_conv_1[1])
            #print (type(self.h1.shape[3]),type(self.all_conv_1[1].shape[1]))
            A_4_input, B_4_input = attention_process_1(A, self.all_conv_1[0], self.all_conv_1[1],'w1_0', 'w1_1')
            A_4_feature,_, _ = Convolutional_Block(A_4_input,64,None,None,TRAIN, 'a1-cnn')
            B_4_feature,_,_ = Convolutional_Block(B_4_input,64,None,None,TRAIN, 'b1-cnn')
            A_1 = distance(A_4_feature, B_4_feature)
            xs1_conv1_aten, xs2_conv1_aten = attention_process_2(A_1, A_4_feature, B_4_feature, 64)
            self.pooled_2_a = tf.squeeze(tf.squeeze(average_pool(xs1_conv1_aten, 'pooled_a4') ,axis=1), axis=1)
            self.pooled_2_b = tf.squeeze(tf.squeeze(average_pool(xs2_conv1_aten, 'pooled_b4') ,axis=1), axis=1)

            pooled_2_a = max_pool(xs1_conv1_aten, conf.filter_size, "pool2_a")
            pooled_2_b = max_pool(xs2_conv1_aten, conf.filter_size, "pool2_b")


            A_2 = distance(pooled_2_a, pooled_2_b)
            A_6_input, B_6_input = attention_process_1(A_2, pooled_2_a, pooled_2_b, 'w2_0', 'w2_1')
            A_6_feature, _, _ = Convolutional_Block(A_6_input,128,None,None,TRAIN,'a2-cnn')
            B_6_feature, _, _ = Convolutional_Block(B_6_input,128,None,None,TRAIN, 'b2-cnn')
            A_3 = distance(A_6_feature, B_6_feature)
            xs1_conv2_aten, xs2_conv2_aten = attention_process_2(A_3, A_6_feature, B_6_feature, 128)
            self.pooled_2_a = tf.squeeze(tf.squeeze(average_pool(xs1_conv2_aten, 'pooled_a4') ,axis=1), axis=1)
            self.pooled_2_b = tf.squeeze(tf.squeeze(average_pool(xs2_conv2_aten, 'pooled_b4') ,axis=1), axis=1)

            #pooled_3_a = max_pool(xs1_conv2_aten, conf.filter_size, "pool3_a")
            #pooled_3_b = max_pool(xs2_conv2_aten, conf.filter_size, "pool3_b")


            '''
            A_4 = distance(pooled_3_a, pooled_3_b)
            A_8_input, B_8_input = attention_process_1(A_4, pooled_3_a, pooled_3_b, 'w3_0', 'w3_1')
            A_8_feature = layer_7_8(A_8_input)
            B_8_feature = layer_7_8(B_8_input)
            A_5 = distance(A_8_feature, B_8_feature)
            xs1_conv3_aten, xs2_conv3_aten = attention_process_2(A_5, A_8_feature, B_8_feature, 512)
            pooled_4_a = average_pool(xs1_conv3_aten, 'pooled_a4')
            pooled_4_b = average_pool(xs2_conv3_aten, 'pooled_b4')

            mix_ab = tf.concat([tf.squeeze(tf.squeeze(pooled_4_a, axis=1), axis=1),
                                     tf.squeeze(tf.squeeze(pooled_4_b, axis=1), axis=1)], axis=0)

            combine_d_xy = tf.concat([mix_ab, self.d_y], 1) # shuffle the data from
            combine_d_xy_shuffle = tf.random_shuffle(combine_d_xy)
            self.x_domain = combine_d_xy_shuffle[:, :-1]
            self.y_domain = combine_d_xy_shuffle[:, -1]
            print('get the mix feature')
            '''
    def adversarial_loss(self, feature, task_label, num_class):
        feature = flip_gradient(feature)
        if self.training == 1:
            feature = tf.nn.dropout(feature, FLAGS.keep_prob)

        # Map the features to TASK_NUM classes
        logits, loss_l2 = fc_layer(feature, num_class, self.l2_loss, "fc-1-2-3_adv")
        loss_adv = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=task_label, logits=logits))

        return loss_adv, loss_l2


