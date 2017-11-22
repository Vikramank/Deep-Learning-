
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:59:19 2016

@author: viky
"""

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


#input data:
x_input=np.linspace(0,10,100)
y_input=5*x_input+2.5

#model parameters
W = tf.Variable(tf.random_normal([1]), name='weight')
#bias
b = tf.Variable(tf.random_normal([1]), name='bias')

#placeholders
with tf.name_scope('input'):
 X=tf.placeholder(tf.float32, name='InputX')
 Y=tf.placeholder(tf.float32, name='InputY')

#model
with tf.name_scope('model'):
 Y_pred=tf.add(tf.mul(X,W),b)

#loss
with tf.name_scope('loss'):
 loss = tf.reduce_mean(tf.square(Y_pred -Y ))
#training algorithm
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#initializing the variables 
#init = tf.initialize_all_variables() #for TF version < 1.0
init=tf.global_variables_initializer()

#starting the session session 
sess = tf.Session()
sess.run(init)
cost=tf.scalar_summary("loss", loss)

sess.run(init)
epoch=2000



merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/Users/Enkay/Documents/Viky/python/tensorflow/lineartensorboard/', graph=tf.get_default_graph())
for step in xrange(epoch):
    _, c, summary=sess.run([train, loss, merged_summary_op], feed_dict={X: x_input, Y: y_input})
    summary_writer.add_summary(summary,step)
    if step%50==0:
     print c
print "Model paramters:"       
print "Weight:%f" %sess.run(W)
print "bias:%f" %sess.run(b)


'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(0,10,10)
y_data = 2*x_data  + 0.3*np.random.rand(*x_data.shape)

#weight
W = tf.Variable(tf.zeros([1]))
#bias
b = tf.Variable(tf.zeros([1]))
#Model-linear regression
y = W * x_data + b

#cost
loss = tf.reduce_mean(tf.square(y - y_data))
#training algorithm
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#initializing the variables
init = tf.initialize_all_variables()

#starting the session session 
sess = tf.Session()
sess.run(init)

# training the line
for step in xrange(1000):
    sess.run(train)
   

print "Model paramters:"       
print "Weight:%f" %sess.run(W)
print "bias:%f" %sess.run(b)

#test 
x_test=np.linspace(0,10,10)
y_test=sess.run(W)*x_test+sess.run(b)
plt.plot(x_data,y_data,'o', x_test,y_test,"*")
'''
