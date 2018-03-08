import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def conv2d(x,W,s):
    return tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r    

shoes = list_files('.\\test')
shoes_train = list_files('.\\shoesData')

def gen_x(fname):
    ix= Image.open(fname)
    ix = ix.resize((64, 64))
    ix = np.asarray(ix, dtype=np.float32)
    ix= np.ravel(ix)
    return ix

shoes_x = []
shoes_x_train = []

for item in shoes:
    ix = gen_x(item)
    shoes_x.append(ix)

for item in shoes_train:
    ix = gen_x(item)
    shoes_x_train.append(ix)

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 64*64*3])
    keep_prob = tf.placeholder(tf.float32)
    new_saver = tf.train.import_meta_graph('shoes_recog_model-275.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    #sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()

    w1 = graph.get_tensor_by_name('Variable:0')
    b1 = graph.get_tensor_by_name('Variable_1:0')

    w2 = graph.get_tensor_by_name('Variable_2:0')
    b2 = graph.get_tensor_by_name('Variable_3:0')

    w3 = graph.get_tensor_by_name('Variable_4:0')
    b3 = graph.get_tensor_by_name('Variable_5:0')

    w4 = graph.get_tensor_by_name('Variable_6:0')
    b4 = graph.get_tensor_by_name('Variable_7:0')

    wfc1 = graph.get_tensor_by_name('Variable_8:0')
    bfc1 = graph.get_tensor_by_name('Variable_9:0')

    wfc2 = graph.get_tensor_by_name('Variable_10:0')
    bfc2 = graph.get_tensor_by_name('Variable_11:0')

    x_image = tf.reshape(x, [-1,64,64,3])

    h_conv1 = tf.nn.relu(conv2d(x_image, w1, 1) + b1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w2, 1) + b2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w3, 1) + b3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_pool3, w4, 1) + b4)
    h_pool4 = max_pool_2x2(h_conv4)
    h_pool4_flat = tf.reshape(h_pool4, [-1, 4*4*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, wfc1)+bfc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, wfc2) + bfc2

    pred = sess.run(y_conv, feed_dict={x:shoes_x, keep_prob:1.0})

    len_in = len(pred)
    rows = len_in//5
    if len_in%5 != 0:
        rows+=1

    c = 0

    fig = plt.figure(figsize=(5,rows))

    fig.subplots_adjust(wspace=1, hspace=0.2)

    for i,j in zip(pred, shoes_x):
        c+=1

        k = np.divide(j, 255.)
        k = k.reshape(64,64,3)


        ax1 = fig.add_subplot(rows, 5, c)
        
        if i[0] > i[1]:
            ax1.text(10, 10, ' Male ', bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5}, fontsize=10)
        else:
            ax1.text(10, 10, 'Female', bbox={'facecolor':'red', 'alpha':0.5, 'pad':5}, fontsize=10)


        ax1.imshow(k)

        ax1.axis('off')
    

    plt.savefig("answer.pdf",dpi=500)
    os.startfile("answer.pdf")