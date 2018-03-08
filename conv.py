import tensorflow as tf
#import conf
import numpy as np
#import preproc
#import genImage
import time
#import string
#import random
#import os
#import matplotlib.pyplot as plt
import math
#from PIL import Image
import pandas as pd

import os                                                                                                             
from PIL import Image

#directory = os.fsencode('./shoesData/a')

BATCH_SIZE = 75
CONV1_FILTER_WIDTH = 5
CONV1_FILTER_HEIGHT = 5
CONV1_OUT_CH_NUM = 32
CONV2_FILTER_WIDTH = 5
CONV2_FILTER_HEIGHT = 5
CONV2_OUT_CH_NUM = 64
CONV3_FILTER_WIDTH = 3
CONV3_FILTER_HEIGHT = 3
CONV3_OUT_CH_NUM = 128
CONV4_FILTER_WIDTH = 3
CONV4_FILTER_HEIGHT = 3
CONV4_OUT_CH_NUM = 256
FC_FILTER_WIDTH = 4
FC_FILTER_HEIGHT = 4
FULLY_CONNECTED_NUM = 1024
DROP_OUT_PROB = 0.5
TRANING_SET_RATE = 0.7
ITERATION = 300

def beautyCM(cm, ind=['True pos', 'True neg'], cols=['Pred pos', 'Pred neg']):
    return pd.DataFrame(cm, index=ind, columns=cols)

def eval_metrics(sess, x_lst, y_lst):
    total_len = len(x_lst)
    num_batch = total_len//3000 + 1
    last_batch_size = total_len - 3000 * (num_batch-1) 
    
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    true_negative = 0.0

    for i in range(num_batch):
        k = 3000*i
        j = min(k+3000, total_len)

        cm = confusion_matrix_tf.eval(feed_dict={x:x_lst[k:j], y_:y_lst[k:j], keep_prob:1.0})
        #pred = sess.run(prediction, feed_dict={x:x_lst[k:j], keep_prob:1.0})
        #label = y_lst[k:j]

        #tp_tensor = tf.metrics.true_positives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #fp_tensor = tf.metrics.false_positives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #fn_tensor = tf.metrics.false_negatives(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1))
        #tn_tensor = tf.metrics.true_negatives_at_thresholds(labels=tf.argmax(label,1) , predictions=tf.argmax(pred,1), thresholds= [0 for i in range(j-k)])
        #tf.global_variables_initializer().run(session=sess)
        #tf.local_variables_initializer().run(session=sess)


        #tp = sess.run(tp_tensor)
        #fp = sess.run(fp_tensor)
        #fn = sess.run(fn_tensor)
        #tn = sess.run(tn_tensor)

        #print(tp, fp, fn)
        #print(tp[0], fp[0], fn[0])
        #tn = (j-k) - tp[0] - fp[0] - fn[0]


        cm_run = beautyCM(cm)

        #print(cm_run)

        #a = true_positive(cm_run)
        #b = false_positive(cm_run)
        #c = false_negative(cm_run)
        #d = true_negative(cm_run)

        a = cm_run.iat[0,0]
        b = cm_run.iat[0,1]
        c = cm_run.iat[1,0]
        d = cm_run.iat[1,1]

        #print(a,b,c,d)

        true_positive += a
        false_positive += b
        false_negative += c
        true_negative += d
    
    acc =  (true_positive + true_negative)/(total_len)
    prec = (true_positive )/(true_positive+false_positive)
    rec = (true_positive )/(true_positive+false_negative)
    
    return acc, prec, rec

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r    

mens = list_files('.\\shoesData\\forman')
womens = list_files('.\\shoesData\\forwoman')

print(len(mens), len(womens))

'''
def gen_x(fname):
    ix= Image.open(fname)
    ix = ix.resize((64, 64))
    ix = np.asarray(ix)
    ix= np.ravel(ix)
    return ix


total_x = []
total_y = []


for item in mens:
    ix = gen_x(item)
    iy = [1,0]
    iy = np.asarray(iy)
    total_x.append(ix)
    total_y.append(iy)

for item in womens:
    ix= gen_x(item)
    iy = [0,1]
    iy = np.asarray(iy)
    total_x.append(ix)
    total_y.append(iy)


# number of mens shoes in training data: 3519 
# number of womens shoes in training data: 11323


it = len(total_x)//(BATCH_SIZE)

x = tf.placeholder(tf.float32, shape=[None, 64*64*3])

print(x)
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # small noise for symmetry breaking
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # prevent dead neuron 
    return tf.Variable(initial)

def conv2d(x,W,s):
    return tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.device('/gpu:0'):

    W_conv1 = weight_variable([CONV1_FILTER_WIDTH,CONV1_FILTER_HEIGHT,3,CONV1_OUT_CH_NUM])
    b_conv1 = bias_variable([CONV1_OUT_CH_NUM])
    

    x_image = tf.reshape(x, [-1,64,64,3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    print('x_im ', x_image)
    print('w1 ',W_conv1)
    print('b1 ',b_conv1)
    print('hc1 ', h_conv1)
    print('hp1 ', h_pool1)

    W_conv2 = weight_variable([CONV2_FILTER_WIDTH,CONV2_FILTER_HEIGHT,CONV1_OUT_CH_NUM,CONV2_OUT_CH_NUM])
    b_conv2 = bias_variable([CONV2_OUT_CH_NUM])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 1) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    print('w2 ',W_conv2)
    print('b2 ',b_conv2)
    print('hc2 ', h_conv2)
    print('hp2 ', h_pool2)

    W_conv3 = weight_variable([CONV3_FILTER_WIDTH,CONV3_FILTER_HEIGHT,CONV2_OUT_CH_NUM,CONV3_OUT_CH_NUM])
    b_conv3 = bias_variable([CONV3_OUT_CH_NUM])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    print('w3 ',W_conv3)
    print('b3 ',b_conv3)
    print('hc3 ', h_conv3)
    print('hp3 ', h_pool3)

    W_conv4 = weight_variable([CONV4_FILTER_WIDTH,CONV4_FILTER_HEIGHT,CONV3_OUT_CH_NUM, CONV4_OUT_CH_NUM])
    b_conv4 = bias_variable([CONV4_OUT_CH_NUM])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, 1) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    print('w4 ',W_conv4)
    print('b4 ',b_conv4)
    print('hc4 ', h_conv4)
    print('hp4 ', h_pool4)

    W_fc1 = weight_variable([FC_FILTER_WIDTH*FC_FILTER_HEIGHT*CONV4_OUT_CH_NUM, FULLY_CONNECTED_NUM])
    b_fc1 = bias_variable([FULLY_CONNECTED_NUM])

    h_pool4_flat = tf.reshape(h_pool4, [-1, FC_FILTER_WIDTH*FC_FILTER_HEIGHT*CONV4_OUT_CH_NUM])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1)+b_fc1)

    print('wfc1 ',W_fc1)
    print('bfc1 ',b_fc1)
    print('hp4 flat ', h_pool4_flat)
    print('hfc1 ', h_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([FULLY_CONNECTED_NUM,2])
    b_fc2 = bias_variable([2])

    print('hcf1 drop ',h_fc1_drop)
    print('wfc2 ',W_fc2)
    print('bfc2 ', b_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print(y_conv)
    #y_conv = tf.nn.sigmoid(y_conv)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    )

    #loss = tf.reduce_sum(y_*tf.log(y_conv) + (1-y_)*tf.log(y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    confusion_matrix_tf = tf.confusion_matrix(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #prediction = y_conv
    prediction = y_conv
    #labels = tf.argmax(y_,1)
    print('\n-------------------------------------------------')
    start_time = time.time()
    for j in range(ITERATION):
        print('Iteration ({0}/{1}) starts training...'.format((j+1), ITERATION))
        for i in range(it):
            xx = total_x[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
            yy = total_y[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
            train_step.run(feed_dict={x: xx, y_: yy, keep_prob:DROP_OUT_PROB})

        if j % 25 == 0 :
            #p_train = sess.run(prediction, feed_dict={x:total_x, keep_prob:1.0})
            #p_label = tf.argmax(total_y,1)
            f_train_acc, _p, _r = eval_metrics(sess, total_x, total_y)
            #f_train_acc = eval_acc(sess, accuracy, total_x, total_y)
            print('\tIter {0} training accuracy = {1}'.format((j+1), f_train_acc))
            print('\tIter {0} training precision = {1}'.format((j+1), _p))
            print('\tIter {0} training recall = {1}'.format((j+1), _r))
            #saver.save(sess,'.\\shoes_recog_model', global_step=j)

            #print(h_conv1.eval(feed_dict={x: total_x[:1]}))

        #total_x, total_y = shift(total_x, total_y)

    #f_train_acc = sess.run(accuracy, feed_dict={x:total_x, y_:total_y, keep_prob:1.0})
    #msgLst = recordExperimentalResult(sess, msgLst)

'''