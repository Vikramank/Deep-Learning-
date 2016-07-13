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
