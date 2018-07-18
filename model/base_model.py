import os
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.layers as layers
import numpy as np
flags = tf.app.flags
flags.DEFINE_string("logdir", "saved_models/", "where to save the model")
flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
flags.DEFINE_float("lrn_rate", 0.001, "learning rate")
flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")

FLAGS = tf.app.flags.FLAGS

L2_REG = 1e-4

class BaseModel(object):

  def set_saver(self, save_dir):
    '''
    Args:
      save_dir: relative path to FLAGS.logdir
    '''
    # shared between train and valid model instance
    self.saver = tf.train.Saver(var_list=None)
    self.save_dir = os.path.join(FLAGS.logdir, save_dir)
    self.save_path = os.path.join(self.save_dir, "model.ckpt")

  def restore(self, session):
    ckpt = tf.train.get_checkpoint_state(self.save_dir)
    self.saver.restore(session, ckpt.model_checkpoint_path)

  def save(self, session, global_step):
    self.saver.save(session, self.save_path, global_step)


def Convolutional_Block(input_, filter_num, filter1, filter2, train, scope):
    norm = tf.contrib.layers.variance_scaling_initializer()
    filter_shape1 = [3, 1, input_.get_shape()[3], filter_num]

    with tf.variable_scope(scope):
        if filter1 == None:
            filter_1 = tf.get_variable('filter1', filter_shape1, initializer=norm)
        else:
            filter_1 = filter1
        conv1 = tf.nn.conv2d(input_, filter_1, strides=[1, 1, filter_shape1[1], 1], padding="SAME")
        batch_normal1 = tf.layers.batch_normalization(conv1, trainable=train)
        batch_normal1_relu = tf.nn.relu(batch_normal1)
        filter_shape2 = [3, 1, batch_normal1_relu.get_shape()[3], filter_num]
        if filter1 == None:
            filter_2 = tf.get_variable('filter2', filter_shape2, initializer=norm)
        else:
            filter_2 = filter2
        conv2 = tf.nn.conv2d(batch_normal1_relu, filter_2, strides=[1, 1, filter_shape2[1], 1], padding="SAME")
        batch_normal2 = tf.layers.batch_normalization(conv2, trainable=train)
        batch_normal2_relu = tf.nn.relu(batch_normal2)

        return batch_normal2_relu, filter_1, filter_2
def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)


def Conv(input_, filter_1 , strides, train, scope):
    with tf.variable_scope(scope):
      conv = tf.nn.conv2d(input_, filter_1, strides=strides, padding="SAME")
      #add bias

      batch_normal = tf.layers.batch_normalization(conv, trainable=train)
      return batch_normal


def linear(inputs, output_dim,  TRAIN, scope=None, stddev=0.1 ):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [inputs.get_shape().as_list()[1], output_dim], initializer=norm, trainable=TRAIN)
        b = tf.get_variable('b', [output_dim], initializer=const, trainable=TRAIN)
        l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        return tf.matmul(inputs, w) + b, l2_loss


def distance(feature_map_a, feature_map_b):
    feature_map_a = tf.squeeze(feature_map_a, axis=2)
    feature_map_b = tf.squeeze(feature_map_b, axis=2)
    A_matmul = tf.matmul(feature_map_a, feature_map_b, transpose_b=True)
    a_squared = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(feature_map_a),axis=2), axis=2),[1,1,feature_map_b.shape[1]])
    b_squared = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(feature_map_b),axis=2), axis=1),[1,feature_map_a.shape[1],1])
    inside_root = a_squared + (-2 * A_matmul) + b_squared  # distance
    '''
    epsilon = tf.Variable(tf.random_uniform([None, feature_map_a.shape[1], feature_map_b.shape[1]],
                                            sys.float_info.epsilon, sys.float_info.epsilon, dtype=np.float32))
    inside_root = tf.maximum(inside_root, epsilon)  # compare with minimal number
    '''
    denominator = 1.0 + tf.sqrt(inside_root)
    A = tf.transpose(1.0 / denominator,perm=(0, 2, 1))
    return A

def average_pool(conv, name):
  pool = tf.layers.average_pooling2d(conv, pool_size=[conv.get_shape()[1], 1],
                              strides=[conv.get_shape()[1], 1], name=name)
  return pool


def attention_process_1(A, input_A, input_B, name1, name2):
  norm = tf.random_normal_initializer(stddev=0.1)
  WA_0 = tf.get_variable(name1, [input_A.get_shape().as_list()[3],
                                  input_B.get_shape().as_list()[1]], initializer=norm)
  WA_1 = tf.get_variable(name2, [input_B.get_shape().as_list()[3],
                                  input_A.get_shape().as_list()[1]], initializer=norm)
  attention_feature_map_a = tf.einsum('ijk,lk->ijl', tf.transpose(A, perm=[0, 2, 1]), WA_0)
  attention_feature_map_b = tf.einsum('ijk,lk->ijl', A, WA_1)

  A_input = tf.expand_dims(tf.concat([tf.squeeze(input_A, axis=2),
                                        attention_feature_map_a], axis=2), axis=2)
  B_input = tf.expand_dims(tf.concat([tf.squeeze(input_B, axis=2),
                                        attention_feature_map_b], axis=2), axis=2)


  return A_input, B_input

def attention_process_2(A,A_feature,B_feature,feature_map):
  col_wise_sum = tf.reduce_sum(A, axis=1)  # col-wise sum (batchsize, seqlen1)
  row_wise_sum = tf.reduce_sum(A, axis=2)

  xs1_conv1_aten = tf.reshape(tf.multiply(tf.reshape(A_feature, [-1, A_feature.get_shape()[1], feature_map]),
                                          tf.reshape(col_wise_sum, [-1, A_feature.get_shape()[1], 1]), ),
                              [-1, A_feature.get_shape()[1], 1, feature_map])

  xs2_conv1_aten = tf.reshape(tf.multiply(tf.reshape(B_feature, [-1, B_feature.get_shape()[1], feature_map]),
                                          tf.reshape(row_wise_sum, [-1, B_feature.get_shape()[1], 1]), ),
                              [-1, B_feature.get_shape()[1], 1, feature_map])
  return xs1_conv1_aten, xs2_conv1_aten


def max_pool(conv1, max_len, name):
  pools = tf.nn.max_pool(conv1, ksize=[1, max_len, 1, 1], strides=[1, 2, 1, 1],
                 padding='SAME', name=name)

  return pools


def fc_layer(feature, num_class, l2_loss, scope, TRAIN):
  with tf.variable_scope(scope):
    '''
    fc1_out, fc1_loss = linear(feature, 1024, TRAIN, scope='fc1', stddev=0.1)
    l2_loss += fc1_loss
    
    fc2_out, fc2_loss = linear(tf.nn.relu(fc1_out), 1024, TRAIN, scope='fc2', stddev=0.1)
    l2_loss += fc2_loss
    
    fc3_out, fc3_loss = linear(tf.nn.relu(fc2_out), num_class, TRAIN, scope='fc3', stddev=0.1)
    '''
    fc3_out, fc3_loss = linear(tf.nn.relu(feature), num_class, TRAIN, scope='fc3', stddev=0.1)
    l2_loss += fc3_loss
    return fc3_out, l2_loss

def diff_loss(shared_feat, task_feat):
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
      loss_diff = tf.identity(cost)

    return loss_diff
def rnn(inputs, rnn_size):



    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    _, enc_out = tf.nn.dynamic_rnn(cell=cell,
                                  inputs=tf.expand_dims(inputs, 0), dtype=tf.float32,
                                           )
    enc_outputs = tf.concat(enc_out, 1)
    outputs = tf.stack(enc_outputs)
    return outputs

def attention(inputs, inputs_hi, question):


        #inputs = tf.squeeze(inputs, 2)
    input_question = tf.expand_dims(question, axis=1)
    vector_attn = tf.reduce_sum(tf.multiply(inputs, input_question), axis=2)
    attention_weights = tf.transpose(tf.expand_dims(tf.nn.softmax(vector_attn, dim=1), -1), [0, 2, 1])

    weighted_projection = tf.multiply(tf.transpose(inputs_hi, [0, 2, 1]), attention_weights)
    outputs = tf.reduce_sum(weighted_projection, axis=2)


    return attention_weights, outputs

class FlipGradientBuilder(object):
  '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''
  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [ tf.negative(grad) * l]
    
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)
        
    self.num_calls += 1
    return y
    
flip_gradient = FlipGradientBuilder()