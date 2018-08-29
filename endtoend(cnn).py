# coding:utf-8

from genIdCard import *
import numpy as np
import tensorflow as tf

# 定义超参数
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = 18
CHAR_SET_LEN = 10

INIT_DECAY = 0.99  # 指数衰减学习率的衰减系数
LAMBDA = 0.0001  # L2正则化参数
LEARNING_RATE = 0.002  # 学习率
TRAINING_STEPS = 6000000  # 指定训练的总次数


# 训练的过程中一边训练一边生成训练样本，每批生成BATCH_SIZE个样本进行训练，这样做的好处是不会占据大量的内存，而且可以随机产生无穷多的训练样本
def gen_batch(batch_size):
    obj = GenIdCard()
    X = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 生成训练样本
    Y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        image, text, vec = obj.gen_image()
        X[i] = image.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        Y[i] = vec
    return X, Y


def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


with tf.device('/cpu:0'):
    # 前向传播和训练
    # input layer
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    train_phase = tf.placeholder(tf.bool)  # 是否在训练阶段
    keep_prob = tf.placeholder(tf.float32)  # dropout的概率值要用变量，因为训练和测试的值是不一样的，训练的时候它的值小于1.0，但是测试的时候其值必须是1.0，即测试时要用到所有神经元
    #################################################################################################
    # 影响
    # 仅仅是在w变量中添加了一个w_alpha=0.01因子，就能把损失/误差的起步值从5.26降到0.69
    #################################################################################################

    w_alpha = 0.01
    b_alpha = 0.1
    # conv + pool layer1
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b_c1)
    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=0.02),
                       train_phase, scope='bn_1')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # # conv + pool layer2
    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_c2)
    conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                       train_phase, scope='bn_2')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # conv + pool layer3
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_c3)
    conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                       train_phase, scope='bn_3')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # conv + pool layer4
    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv4 = tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_c4)
    conv4 = batch_norm(conv4, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                       train_phase, scope='bn_4')
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # flatten last pooling layer
    lst = conv4.get_shape().as_list()  # lst=[None,2,16,64]
    nodes = lst[1] * lst[2] * lst[3]
    conv4 = tf.reshape(conv4, [-1, nodes])

    # fully connected layer
    w_fc1 = tf.Variable(w_alpha * tf.random_normal([nodes, 1024]))
    b_fc1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    fc1 = tf.matmul(conv4, w_fc1)
    fc1 = tf.add(fc1, b_fc1)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # output layer
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out_u = tf.add(tf.matmul(fc1, w_out), b_out)

    # 期望输出，输出数据是18*10的一维数组
    t = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])

    # 损失函数
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_u, labels=t))

    # 添加L2正则化得到最终的损失函数
    # regularizer = tf.contrib.layers.l2_regularizer(LAMBDA)
    # regularization = regularizer(w_c1) + regularizer(w_c2) + regularizer(w_c3) + regularizer(w_c4) + regularizer(
    #     w_fc1) + regularizer(w_out)
    # loss = cross_entropy + regularization

    # 添加指数衰减学习率的优化方法的训练
    # global_step = tf.Variable(0, trainable=False)   # 因为后面会计算张量train_step计算过程中所有涉及到的变量作为更新对象，而不需要更新的变量就得设置trainable=False
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, TRAINING_STEPS / BATCH_SIZE, INIT_DECAY)
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

    # 精确度计算，out_u和t的形状是[None,180]
    out_u_temp = tf.reshape(out_u, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    t_temp = tf.reshape(t, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    prediction1 = tf.equal(tf.argmax(out_u_temp, 2),
                           tf.argmax(t_temp, 2))  # 计算batch中所有数字的精确度 prediction1形状为[-1,18] 值为True表示预测正确的数字
    accuracy1 = tf.reduce_mean(tf.cast(prediction1, tf.float32))

    prediction2 = tf.reduce_min(tf.cast(prediction1, tf.float32), -1)  # 计算batch中所有样本的精确度，只有一个样本中所有18个数字都预测正确，该样本才是预测正确的
    accuracy2 = tf.reduce_mean(prediction2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    import time
    for i in range(TRAINING_STEPS):
        # start = time.time()
        X, Y = gen_batch(50)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: X, t: Y, keep_prob: 0.75, train_phase: True})  # 把每次的输入和期望输出喂给系统
        print(i, loss)
        # end = time.time()
        # print("耗时：", end - start)
        if i % 100 == 0 and i != 0:  # 每迭代100次计算一次准确率
            X, Y = gen_batch(50)
            acc1, acc2 = sess.run([accuracy1, accuracy2], feed_dict={x: X, t: Y, keep_prob: 1.0, train_phase: False}) # 之前这里keep_prob的值忘了改为1.0了，折腾了一下午才找到这个错误
            print("第%s步，准确率1为：%s，准确率2为：%s" % (i, acc1, acc2))
            if acc1 > 0.9:
                saver.save(sess, "/home/apex/projects/endtoend/model", global_step=i)
