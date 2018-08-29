import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_data(size=100000):
    """
    生成数据:
        输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
        输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
    """
    X = np.random.choice(2, (size,))
    Y = []
    for i in range(size):
        threshold = 0.5
        # 判断X[i-3]和X[i-8]是否为1，修改阈值
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        # 生成随机数，以threshold为阈值给Yi赋值
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    # raw_data是使用gen_data()函数生成的数据，分别是X和Y
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)

    # 因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。
    for i in range(batch_size):
        data_x[i] = raw_x[i * batch_partition_length:(i + 1) * batch_partition_length]
        data_y[i] = raw_y[i * batch_partition_length:(i + 1) * batch_partition_length]

    # x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    epoch_size = batch_partition_length // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    '''这里的n就是训练过程中用的epoch，即在样本规模上循环的次数'''
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps=num_steps)


batch_size = 5
num_steps = 10      # RNN中重复DNN的次数
state_size = 10     # 隐藏层节点数
n_classes = 2       # 分类数
learning_rate = 0.1

x = tf.placeholder(tf.int32, [batch_size, num_steps])
y = tf.placeholder(tf.int32, [batch_size, num_steps])

x_one_hot = tf.one_hot(x, n_classes)                # 将输入转化为one-hot编码，[batch_size, num_steps, num_classes]
rnn_inputs = tf.unstack(x_one_hot, axis=1)          # 在num_steps上解绑，方便给每个循环单元输入,RNN的输入就是one-hot编码的序列值
y_as_lists = tf.unstack(y, axis=1)                  # Turn our y placeholder into a list of labels
# 输入和输出都在num_steps上解绑，这样就是面向单个全连接神经网络进行输入和输出的，这里有num_steps个全连接神经网络

# 隐藏层前向传播
w1 = tf.Variable(tf.random_normal([n_classes + state_size, state_size]))    # 定义隐藏层的权值
b1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[state_size]))    # 定义隐藏层的偏置
a1 = []                                                                     # 定义隐藏层的输出
state = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[batch_size, state_size]), trainable=False)    # 初始化state（表示上一次的输出值）

# 循环num_steps次，将一个序列输入RNN模型
for rnn_input in rnn_inputs:
    state = tf.tanh(tf.matmul(tf.concat((rnn_input, state), 1), w1) + b1)
    a1.append(state)
final_state = a1[-1]


# 输出层前向传播
w2 = tf.Variable(tf.random_normal([state_size, n_classes]))                  # 定义输出层的权值
b2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))      # 定义输出层的偏置
u2 = [tf.matmul(a, w2) + b2 for a in a1]                                     # 定义输出层的输入总和，用列表来包括num_steps个全连接神经网络的计算结果

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for label, logit in zip(y_as_lists, u2)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


# 使用动态rnn时改为下面的代码
# a1 = tf.reshape(a1, [-1, state_size])   # a1原来的形状是(num_steps, batch_size, state_size)
# u2 = tf.matmul(a1, w2) + b2     # 因为输出层没有state输入，num_steps个全连接神经网络共用一份权值和偏置，所以这里把batch_size和num_steps两个维度压缩在一起，只执行一步“乘以权值再加偏置”
# u2 = tf.reshape(u2, [batch_size, num_steps, n_classes])     # 然后再把它的形状还原回来
#
# losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=u2)
# total_loss = tf.reduce_mean(losses)
# train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

def train_network(num_epochs, num_steps, state_size, verbose=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        # 得到数据
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            # 保存每次执行后的最后状态，然后赋给下一次执行
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print('\EPOCH', idx)
            # 这是具体获得数据的部分
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                             feed_dict={x: X, y: Y, state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 100 steps:", training_loss / 100)
                    training_losses.append(training_loss / 100)
                    training_loss = 0

            return training_losses


training_losses = train_network(1, num_steps, state_size)
plt.plot(training_losses)
plt.show()
