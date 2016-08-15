# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 10:03:15 2016

@author:Viky
Code for polynomial regression
"""

#importing necessary packages
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt


#input data:
x_input=np.linspace(0,3,1000)
x1=x_input/np.max(x_input)
x2=np.power(x_input,2)/np.max(np.power(x_input,2))
y_input=5*x1-3*x2
y_input= y_input.reshape((y_input.size, 1))

#model parameters
#order of polynomial 
n=2
W = tf.Variable(tf.random_normal([n,1]), name='weight')
#bias
b = tf.Variable(tf.random_normal([1]), name='bias')

#X=tf.placeholder(tf.float32,shape=(None,2))
X=tf.placeholder(tf.float32,shape=[None,n])
Y=tf.placeholder(tf.float32,shape=[None, 1])


#preparing the data
def modify_input(x,x_size,n_value):
   x_new=np.zeros([x_size,n_value])    
   for i in range(n):
      x_new[:,i]=np.power(x,(i+1))
      x_new[:,i]=x_new[:,i]/np.max(x_new[:,i])
   return x_new
 

#model
x_modified=modify_input(x_input,x_input.size,n)
Y_pred=tf.add(tf.matmul(X,W),b)

#algortihm
loss = tf.reduce_mean(tf.square(Y_pred -Y ))
#training algorithm
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#initializing the variables
init = tf.initialize_all_variables()

#starting the session session 
sess = tf.Session()
sess.run(init)

epoch=12000

for step in xrange(epoch):     
     _, c=sess.run([optimizer, loss], feed_dict={X: x_modified, Y: y_input})
     if step%1000==0 :
       print c

print "Model paramters:"       
print  sess.run(W)
print "bias:%f" %sess.run(b)
# comparing our model 
y_test=sess.run(Y_pred, feed_dict={X:x_modified})
plt.plot(x_input,y_input,x_input, y_test)
plt.show()