#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tf CNN+LSTM+CTC 训练识别不定长数字字符图片
@author: pengyuanjie
"""
from genIdCard import *
import numpy as np
import tqdm
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义一些常量
INPUT_SHAPE = (32, 256)  # 图片大小，32 x 256
num_epochs = 2000  # 训练最大轮次
num_hidden = 64
num_layers = 1
num_classes = GenIdCard().len + 1 + 1  # 10位数字 + blank + ctc blank
INITIAL_LEARNING_RATE = 1e-3  # 学习率
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9
DIGITS = '0123456789'
BATCHES = 10
BATCH_SIZE = 30
TRAIN_SIZE = BATCHES * BATCH_SIZE


def decode_sparse_tensor(sparse_tensor):
    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        # print(result)
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = GenIdCard()
    inputs = np.zeros([batch_size, INPUT_SHAPE[1], INPUT_SHAPE[0]])  # (batch_size,256,32)
    codes = []

    for i in range(batch_size):
        image, text, vec = obj.gen_image(True)  # 生成不定长度的字串
        inputs[i] = np.transpose(image.reshape((INPUT_SHAPE[0], INPUT_SHAPE[1])))  # 将转置为(256,32)的image赋值给inputs
        codes.append(list(text))
        # codes是batch_size个元素的列表[['0', '2', '5', '7', '3', '4', '0', '5'], ['9', '8', '6', '6']]
    targets = [np.asarray(i) for i in codes]
    # targets是元素数组化的codes [array(['0', '2', '5', '7', '3', '4', '0', '5'], dtype='<U1'), array(['9', '8', '6', '6'], dtype='<U1')]
    sparse_targets = sparse_tuple_from(targets)
    seq_len = np.ones(batch_size) * INPUT_SHAPE[1]  # 将np.ones()数组中的元素乘以INPUT_SHAPE[1]
    import pdb
    # pdb.set_trace()
    return inputs, sparse_targets, seq_len
    # inputs是(2, 256, 32)，
    # sparse_targets是(array([
    #        [0, 0],
    #        [0, 1],
    #        [0, 2],
    #        [0, 3],
    #        [0, 4],
    #        [0, 5],
    #        [0, 6],
    #        [0, 7],
    #        [1, 0],
    #        [1, 1],
    #        [1, 2],
    #        [1, 3]], dtype=int64), array([0, 2, 5, 7, 3, 4, 0, 5, 9, 8, 6, 6]), array([2, 8], dtype=int64))
    # seq_len是 array([256., 256.])


# 定义CNN网络，处理图片，
def convolutional_layers():
    # 输入数据，shape [batch_size, max_stepsize, num_features]
    inputs = tf.placeholder(tf.float32, [None, None, INPUT_SHAPE[0]])

    # 第一层卷积层, 32*256*1 => 16*128*48
    w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 48], stddev=0.5))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[48]))
    x_expanded = tf.expand_dims(inputs, 3)  # 给inputs添加一个channel维度
    h_conv1 = tf.nn.conv2d(x_expanded, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(h_conv1 + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二层, 16*128*48 => 16*64*64
    w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 48, 64], stddev=0.5))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2 = tf.nn.relu(h_conv2 + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    # 第三层, 16*64*64 => 8*32*128
    w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.5))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3 = tf.nn.relu(h_conv3 + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    # 平化最后一层
    # conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 8 * INPUT_SHAPE[1]])
    conv_layer_flat = tf.reshape(h_pool3, [-1, 8 * 32 * 128])

    # 全连接
    w_fc1 = tf.Variable(tf.truncated_normal([8 * 32 * 128, INPUT_SHAPE[1]], stddev=0.5))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[INPUT_SHAPE[1]]))
    features = tf.nn.relu(tf.matmul(conv_layer_flat, w_fc1) + b_fc1)
    shape = tf.shape(features)  # （batchsize,256）
    features = tf.reshape(features, [shape[0], INPUT_SHAPE[1], 1])  # （batchsize,256,1）
    return inputs, features


def get_train_model():
    # features = convolutional_layers()
    # print features.get_shape()

    inputs = tf.placeholder(tf.float32, [None, None, INPUT_SHAPE[0]])       # lstm的输入的width不受限制
    targets = tf.sparse_placeholder(tf.int32)  # 定义ctc_loss需要的稀疏矩阵
    seq_len = tf.placeholder(tf.int32, [None])  # 1维向量 序列长度 [batch_size,]

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  # 创建num_layers==1个隐藏层
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)  #
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs, [-1, num_hidden])
    w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    logits = tf.matmul(outputs, w) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits, inputs, targets, seq_len, w, b


def train():
    inputs = tf.placeholder(tf.float32, [None, None, INPUT_SHAPE[0]])  # num_steps竟然是图片的宽,而这里图片的宽可以不定，而每个DNN的输入是图片的高32
    targets = tf.sparse_placeholder(tf.int32)  # 定义ctc_loss需要的稀疏矩阵
    seq_len = tf.placeholder(tf.int32, [None])  # 1维向量 序列长度 形状是[batch_size,]

    # 定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    outputs = tf.reshape(outputs, [-1, num_hidden])
    w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")  # 定义输出层权值
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")  # 定义输出层偏置
    u = tf.matmul(outputs, w) + b
    u = tf.reshape(u, [BATCH_SIZE, -1, num_classes])
    u = tf.transpose(u, (1, 0, 2))  # 转置为[num_steps, batch_size, num_classes]

    loss = tf.nn.ctc_loss(labels=targets, inputs=u, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)     # 这里将loss或cost传入优化器都可以
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # 计算精确度
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(u, seq_len, merge_repeated=False)
    t = decoded[0]
    distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, ".\\model\\model")
        except ValueError:
            print("model does not exist!")

        sess.run(tf.global_variables_initializer())
        for i in tqdm.tqdm(range(num_epochs)):
            inputs1, targets1, seq_len1 = get_next_batch(BATCH_SIZE)
            sess.run([optimizer], feed_dict={inputs: inputs1, targets: targets1, seq_len: seq_len1})
        print("train complete! saving model.....")
        saver.save(sess, ".\\model\\model")         # 训练完毕，保存模型

        # 测试模型
        inputs1, targets1, seq_len1 = get_next_batch(BATCH_SIZE)
        decoded, distance = sess.run([t, distance], feed_dict={inputs: inputs1, targets: targets1, seq_len: seq_len1})
        print("acc", distance)
        print(decoded.values)
        print(targets1[1])
        # for i in decoded.values:
        #     print(i)
        #     if i==j:
        #         print("True",i,j)
        #     else:
        #         print("False")

        # 只能运行一次，否则报错：Fetch argument 12.613963 has invalid type <class 'numpy.float32'>, must be a string or Tensor. (Can not convert a float32 into a Tensor or Operation.)


if __name__ == '__main__':
    # inputs, sparse_targets, seq_len = get_next_batch(2)
    # decode_sparse_tensor(sparse_targets);
    train()
