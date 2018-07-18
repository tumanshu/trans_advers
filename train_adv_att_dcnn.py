# coding=utf-8
#! /usr/bin/env python

import tensorflow as tf
import os
import time
import codecs
from input import data_helper
import numpy as np
from model.atcnn import atCNN
from model.Char_CNN import CharCNN
import datetime
from model.base_model import diff_loss
from model.q_nn import q_nn
from model.layer_att import HAN
# Parameters
# ==================================================
# Data loading params
flags = tf.flags

tf.flags.DEFINE_string("A_path", "books.task", "Data source for the source path")
tf.flags.DEFINE_string("B_path", "dvd.task", "Data target for the target path")
tf.flags.DEFINE_string("domain_list", "books,dvd", "domain Data list for domain train")
tf.flags.DEFINE_string("dir_path", "data", "Data source for the target path")
tf.flags.DEFINE_boolean("word_vec", True, "if have word2vec")
flags.DEFINE_string("word2vec_path", "word2vec.npy", "the word2vec path")
#flags.DEFINE_integer("max_char_length_d", 500, "the word max sentence char length")
#flags.DEFINE_integer("max_char_length_s", 500, "the word max sentence char length")

#the cnn params
flags.DEFINE_integer("embedding_size", 300, "The converlutional filter size")
flags.DEFINE_integer("feature_map", 300, "The converlutional filter size")
flags.DEFINE_integer("question_length", 100, "The question word length")
flags.DEFINE_integer("filter_size", 3, "The converlutional filter size")
flags.DEFINE_integer("pooling_size", 9, "The pooling size")
flags.DEFINE_integer("k_max_value", 8, "The k_max pooling value (default :3) ")
#dictionary para

flags.DEFINE_integer("k_max_word", 8, "Get k_max sentiment word")
flags.DEFINE_string("dict_path", "dict_data", "Get k_max sentiment word")
flags.DEFINE_string("domain_dict", "all_sentiment_dict", "Get k_max sentiment word")
flags.DEFINE_string("A_dict", "dict_data_books", "Get k_max sentiment word")
flags.DEFINE_string("B_dict", "dict_data_dvd", "Get k_max sentiment word")
flags.DEFINE_string("senti_word2vec_path", "word2vec_se.npy", "the word2vec path")

#rnn params
flags.DEFINE_integer('rnn_size', 150, 'RNN unit size') #150,200,250
flags.DEFINE_integer('word_attention_size', 100, "word attention size")#100,150,200
flags.DEFINE_integer('sent_attention_size', 100, "sent_attention_size")
#input params
flags.DEFINE_integer('document_size', 30, 'document size')
flags.DEFINE_integer('sentence_size', 20, 'sentence size')
flags.DEFINE_integer('con_or_sen', False, 'sentence size')

flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_integer("domain_class", 2, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_integer("sentiment_class", 2, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 5, "Batch Size (default: 64)")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")
flags.DEFINE_integer("num_epochs", 120, "Number of training epochs (default: 120)")
flags.DEFINE_float("initial_learning_rate", 0.01, "Number of training epochs (default: 200)")
flags.DEFINE_integer("train_num", 20, "Number of training epochs (default: 200)")
flags.DEFINE_integer("dev_num", 50, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

conf = tf.flags.FLAGS
dir_path = conf.dir_path
dict_path = conf.dict_path

print("\nParameters:")
for attr, value in sorted(conf.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


if conf.word_vec:
    word2vec = np.load(conf.word2vec_path)
    word2vec_s = np.load(conf.senti_word2vec_path)
else:
    word2vec = None
    word2vec_s = None
# Data Preparation
# ==================================================
print("Loading dic...")
#DIC = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}\t"

DIC = data_helper.readDic(os.path.join(dict_path,"dict_number"))
#cdict = {}
#for i,c in enumerate(DIC):
#    cdict[c] = i
dict_all=dict(**DIC)

all_sentiment_dic = data_helper.readse_idx(os.path.join(dict_path,"all_sentiment_dict"), dict_all)
A_dict = data_helper.readse_idx(os.path.join(dict_path, conf.A_dict), dict_all)
B_dict = data_helper.readse_idx(os.path.join(dict_path, conf.B_dict), dict_all)
A_B = list(set(A_dict).intersection(set(list(B_dict))))
B_dict = list(set(B_dict).difference(set(list(A_B))))
all_sentiment_dic.extend(A_B)



print("Loading data...")
#add all data used for domian classifier
d_x, d_y, num_d, conf.domain_class = data_helper.make_idx_data_domain(dir_path, conf.domain_list,
                                                                      conf.document_size*conf.sentence_size,
                                                                      dict_all, conf.batch_size)

s_c_x, s_c_y, num_s_s, s_c_x_set, sent_len_s, doc_len_s = data_helper.make_idx_data_sentiment(os.path.join(dir_path, conf.A_path + ".train"),
                                                            conf, dict_all)
s_t_x, s_t_y, num_s_t, s_t_x_set, sent_len_t, doc_len_t = data_helper.make_idx_data_sentiment(os.path.join(dir_path, conf.B_path + ".test"),
                                                            conf, dict_all)

#d_x, d_y = data_helper.shuffled((num_d), d_x, d_y)
se_t_x, se_t_y, se_t_set = data_helper.shuffled(num_s_t, s_t_x, s_t_y, s_t_x_set)

se_x, se_y, se_x_set = data_helper.shuffled(len(s_c_y), s_c_x, s_c_y, s_c_x_set)
'''
se_x = s_c_x
se_x_set = s_c_x_set
se_y = s_c_y
'''
#se_x, se_y = data_helper.copy_data((num_d), se_x, se_y)

print("number of source and target train data: " + str(num_d))
print("number of train sentiment data:" + str(se_y.shape))
print("number of dev sentiment data:" + str(num_s_t))
print("loading over")

# Training
# ==================================================
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth= False)
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=conf.allow_soft_placement,
        log_device_placement=conf.log_device_placement)
    sess = tf.Session(config=session_conf)
    with tf.device('/gpu:0') and sess.as_default():

        domain_model = q_nn(conf, len(dict_all), word2vec, word2vec_s)
        sentiment_feature = CharCNN(conf, domain_model, all_sentiment_dic, B_dict)
        #atcnn = atCNN(conf, len(dict_all))
        loss_se, l2_loss_se = sentiment_feature.se_loss(sentiment_feature.h2, conf.sentiment_class)
        #loss_adv, l2_loss_adv = domain_model.adversarial_loss(domain_model.pooled1,
        #                                                      conf.domain_class)
        #loss_diff = diff_loss(sentiment_feature.sent_attn_outputs, domain_model.pooled1)
        l2_loss = conf.l2_reg_lambda*(l2_loss_se)#+l2_loss_adv)
        loss_all = loss_se+l2_loss#+loss_adv+loss_diff
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
        #rate_ = tf.div(0.0075,tf.sqrt(tf.sqrt(tf.pow(tf.add(1.0,tf.div(global_step,120.0)), 3.0))))
        '''
        learning_rate = tf.train.exponential_decay(conf.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10, decay_rate=0.95)
        '''
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        optimizer = tf.train.AdamOptimizer(conf.initial_learning_rate)  # tf.train.AdamOptimizer(0.0001) #
        grads_and_vars = optimizer.compute_gradients(loss_all)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs" + "_" + conf.dir_path, timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf.num_checkpoints)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(d_x_train, d_y_train, se_x, se_x_set, se_y, sent_len_s, doc_len_s, flags):
            """
            A single training step
            """
            feed_dict = {
                domain_model.input_x_d: d_x_train,
                domain_model.d_y: d_y_train,
                sentiment_feature.input_x_se: se_x,
                sentiment_feature.input_x_set: se_x_set,
                #sentiment_feature.sentence_len:sent_len_s,
                #sentiment_feature.doc_len:doc_len_s,
                sentiment_feature.se_y: se_y,
                domain_model.training: 1,
                sentiment_feature.training: 1,
                domain_model.keep_prob: conf.keep_prob,
                sentiment_feature.keep_prob: conf.keep_prob
            }

            emb_s, emb_d, score,  word_set, indicate, indicater, all_score, all_score_no, word_dict = sess.run([
                                                    sentiment_feature.embedded_characters_set,
                                                    sentiment_feature.embedded_dict,
                                                    sentiment_feature.max_value,
                                                    sentiment_feature.input_max,
                                                    sentiment_feature.max_indicate,
                                                    sentiment_feature.max_indicaters,
                                                    sentiment_feature.input_word_sentiment,
                                                    sentiment_feature.input_word_sentiment_mul,
                                                    sentiment_feature.input_max_id],
                                                                   feed_dict)

            np.savetxt("emb_s", emb_s[0])
            np.savetxt("emb_d", emb_d)
            #print(score, train_data_domian_y, indicate)


            for i in word_set:
                for j in i:
                    for key,value in dict_all.items():
                        if value == j:
                            print (key)

            for i in word_dict:
                for j in i:
                    for key,value in dict_all.items():
                        if value == j:
                            print (key)


            _, step, loss_se_, accuracy_se= sess.run(
                [train_op, global_step, loss_se, sentiment_feature.accuracy],
                feed_dict) # ,, loss_adv_,  accuracy_d,  loss_adv,domain_model.accuracy_adv
            time_str = datetime.datetime.now().isoformat()
            if flags:
                print("{}: step {}, loss_se {:g},  acc_se {:g}"
                      "".format(time_str, step, loss_se_, accuracy_se)) #,,loss_adv_,accuracy_d,, loss_adv{:g},acc_d {:g}

        def dev_step(se_t_x, se_t_set, se_t_y, sent_len_t, doc_len_t):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                domain_model.input_x_d: se_t_x,
                sentiment_feature.input_x_se: se_t_x,
                sentiment_feature.input_x_set: se_t_set,
                #sentiment_feature.sentence_len: sent_len_t,
                #sentiment_feature.doc_len: doc_len_t,
                sentiment_feature.se_y: se_t_y,
                domain_model.training: 0,
                sentiment_feature.training: 0,
                domain_model.keep_prob: 1,
                sentiment_feature.keep_prob: 1,

            }
            #test_data_domian_x, test_data_domian_y = sess.run([sentiment_feature.input_x_se, sentiment_feature.se_y],
            #                                                    feed_dict)
            #print(test_data_domian_x, test_data_domian_y)
            step,  accuracy, loss_test = sess.run(
                [global_step, sentiment_feature.accuracy, loss_se],
                feed_dict)

            return accuracy,step, loss_test


        # Generate batches

        write_path = os.path.join(checkpoint_prefix, conf.A_path+" "+conf.B_path+".txt")
        if not os.path.exists(write_path):
            os.mkdir(checkpoint_prefix)
        w_ = codecs.open(write_path, "w")
        w_.write("sentence_size:"+str(conf.sentence_size)+"\r\n")
        w_.write("document_size:"+str(conf.document_size)+"\r\n")
        w_.write("k_max_word:"+str(conf.k_max_word)+"\r\n")

        batches_train = data_helper.batch_iter(
            list(zip(d_x, d_y, se_x, se_x_set, se_y, sent_len_s, doc_len_s)), conf.batch_size, conf.num_epochs, shuffle=True)

        # Training loop. For each batch...
        i = 0
        for batch in batches_train:
            d_x, d_y, se_x, se_x_set, se_y, sent_len_s, doc_len_s = zip(*batch)

            if i == conf.train_num:
                train_step(d_x, d_y, se_x, se_x_set, se_y, sent_len_s, doc_len_s,  flags=True)
                i = 0
            else:
                train_step(d_x, d_y, se_x, se_x_set, se_y, sent_len_s, doc_len_s, flags=False)
                i = i + 1
            current_step = tf.train.global_step(sess, global_step)
            loss_all = []
            acc_all = []
            final_loss = 0
            final_acc = 0
            if current_step % conf.dev_num == 0:
                batches_dev = data_helper.batch_iter(
                    list(zip(se_t_x, se_t_set, se_t_y, sent_len_t, doc_len_s)), conf.batch_size, 1)
                print("\nEvaluation:")
                for bat in batches_dev:
                    x_bat, x_set, y_bat, sent_len_t, doc_len_t = zip(*bat)

                    # sentence_length_dev_list.reverse()
                    acc, step, loss_step = dev_step(x_bat, x_set, y_bat, sent_len_t, doc_len_t)
                    acc_all.append(acc)
                    loss_all.append(loss_step)
                time_str = datetime.datetime.now().isoformat()
                for acc in acc_all:
                    final_acc += acc
                for loss in loss_all:
                    final_loss += loss
                w_.write(str(final_acc/len(acc_all))+"\r\n")
                print("{}: acc {:g}".format(time_str,final_acc / len(acc_all)))
                print("{}: loss {:g}".format(time_str,final_loss / len(loss_all)))
            if current_step % conf.dev_num == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        w_.close()






