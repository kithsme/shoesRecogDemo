import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import (manifold, random_projection)
import random
from matplotlib import offsetbox
from sklearn.neighbors import NearestNeighbors


        
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

shoes_name_lst = []
for i in shoes:
    g = i.split('\\')
    shoes_name_lst.append(g[-1])

shoes_name_set = set([])

shoes_train_full = list_files('.\\shoesData')
shoes_train_dup_remved = []
target=[]
for i in shoes_train_full:
    g = i.split('\\')

    if g[-1] in shoes_name_set:
        pass
    else:
        shoes_name_set.add(g[-1])
        shoes_train_dup_remved.append(i)
        if 'forman' in g:
            target.append(1)
        elif 'forwoman' in g:
            target.append(2)
        else:
            print('????????? this should not happen')

shoes_train = shoes_train_dup_remved


def plot_embedding(X, y, figs, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 9e-3:
                #don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            k = np.divide(figs[i], 255.)
            k = k.reshape(64,64,3)

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(k),
                X[i])
            ax.add_artist(imagebox)
            
    plt.axis('off')
    if title is not None:
        plt.title(title)
    

    plt.savefig("answer.pdf",dpi=500)
    os.startfile("answer.pdf")
    plt.show()


def gen_x(fname):
    ix= Image.open(fname)
    ix = ix.resize((64, 64))
    ix = np.asarray(ix, dtype=np.float32)
    ix= np.ravel(ix)
    return ix

#shoes_x = []
shoes_x_train = []


#for item in shoes:
#    ix = gen_x(item)
#    shoes_x.append(ix)

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
    #y_conv = tf.matmul(h_fc1_drop, wfc2) + bfc2

    #vec = sess.run(h_fc1_drop, feed_dict={x:shoes_x, keep_prob:1.0})

    #print(type(vec))
    vec_train1 = sess.run(h_fc1_drop, feed_dict={x:shoes_x_train[:3000], keep_prob:1.0})
    vec_train2 = sess.run(h_fc1_drop, feed_dict={x:shoes_x_train[3000:6000], keep_prob:1.0})
    vec_train3 = sess.run(h_fc1_drop, feed_dict={x:shoes_x_train[6000:9000], keep_prob:1.0})
    vec_train4 = sess.run(h_fc1_drop, feed_dict={x:shoes_x_train[9000:12000], keep_prob:1.0})
    vec_train5 = sess.run(h_fc1_drop, feed_dict={x:shoes_x_train[12000:], keep_prob:1.0})

    vec_train = np.concatenate((vec_train1, vec_train2, vec_train3, vec_train4, vec_train5))
    
    randix = np.random.randint(len(vec_train), size=len(vec_train)//10)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    print('start')
    X_tsne = tsne.fit_transform(vec_train[randix,:])
    print('end')
    plot_embedding(X_tsne, [target[a] for a in randix],  [shoes_x_train[a] for a in randix])

