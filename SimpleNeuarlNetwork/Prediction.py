# encoding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import random


def showImg(image):
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

random = random.randint(0, 2000)

test = mnist.test.images[random]

hidden_layers = 100

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
label = tf.argmax(logits, 1)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./Model/model.ckpt")
    print(sess.run(label, feed_dict={input_data: [test]}))

test = np.reshape(test, (28, 28))
showImg(test)
