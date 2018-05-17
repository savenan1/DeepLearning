# encoding=utf-8

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 获取数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 超参数设置
epochs = 20
batch_size = 128
hidden_layers = 100
learningRate = 0.01

# 创建入参占位符
input_data = tf.placeholder(tf.float32, [None, 784], name="input_data")
outputs = tf.placeholder(tf.float32, [None, 10], name="output_data")
learning_rate = tf.placeholder(tf.float32, name="learning_rate")

# 构建隐层
w1 = tf.Variable(tf.truncated_normal([784, hidden_layers], stddev=0.1), name="w1")
b1 = tf.Variable(tf.zeros(hidden_layers), name="b1")

hidden_layer = tf.matmul(input_data, w1) + b1
hidden_layer = tf.nn.sigmoid(hidden_layer)

# 构建输出层
w2 = tf.Variable(tf.truncated_normal([hidden_layers, 10], stddev=0.1), name="w2")
b2 = tf.Variable(tf.zeros(10), name="b2")

logits = tf.matmul(hidden_layer, w2) + b2
logits = tf.nn.softmax(logits)

# 计算误差
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
loss = tf.reduce_mean(loss)

# 反向传播优化网络w和b
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# 用于保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for j in range(mnist.train.num_examples // 128):
            train_data = mnist.train.next_batch(batch_size)
            # print(train_data[0])
            # print(train_data[1])
            # exit(0)

            sess.run(optimizer,
                     feed_dict={input_data: train_data[0], outputs: train_data[1], learning_rate: learningRate})

        cost = sess.run(loss, feed_dict={input_data: train_data[0], outputs: train_data[1]})
        trainAcc = sess.run(accuracy, feed_dict={input_data: train_data[0], outputs: train_data[1]})
        validAcc = sess.run(accuracy, feed_dict={input_data: mnist.validation.images, outputs: mnist.validation.labels})

        print("epoch:", i)
        print("cost", cost)
        print("train accuracy", trainAcc)
        print("valid accuracy", validAcc)
    saver.save(sess, "Model/model.ckpt")
print("finish..............")



