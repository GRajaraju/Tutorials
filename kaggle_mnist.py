
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

# Importing Kaggle MNIST data
data = np.loadtxt(open('/train.csv','rb'), delimiter=",", skiprows=1)
y_labels = data[:,0]
x_data = data[:,1:]
y_labels = y_labels.reshape((42000,1))

# Converting labels into one-hot encoder
label_size = 10
y_data = []
for i in range(len(y_labels)):
    idx = int(y_labels[i])
    y_temp = np.zeros((label_size))
    y_temp[idx] = 1
    y_data.append(y_temp)
    
y_data = np.array(y_data)

# Creating placeholders for a and y
X = tf.placeholder(tf.float32,[None,784], name='X')
Y = tf.placeholder(tf.float32, [None,10], name='Y')

# Initialising weights and biases
W1 = tf.Variable(tf.random_normal([784,300], stddev=0.01), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
W2 = tf.Variable(tf.random_normal([300,10],stddev=0.01), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

learning_rate = 0.01
epochs = 10
batch_size = 100

# function to create training batches for X and Y
def batch_data(x_data,y_data,batch_size):
    i = 0
    start = 0
    num_batches = int(x_data.shape[0] / batch_size)
    while i < num_batches:
        x_batch = x_data[start:start+batch_size,:]
        y_batch = y_data[start:start+batch_size,:]
        start += batch_size
        i += 1
        yield x_batch, y_batch

hidden1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
y_hat = tf.nn.softmax(tf.add(tf.matmul(hidden1,W2),b2))

y_clipped = tf.clip_by_value(y_hat, 1e-10,0.9999999)

# Loss function
loss = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(y_clipped) + (1 - Y) * tf.log(1 - y_clipped), axis=1))

# Adding optimiser
training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Initialising global variables
init = tf.global_variables_initializer()

# Accuracy Check
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Start the session
with tf.Session() as sess:
    sess.run(init)
    
    total_batch = int(x_data.shape[0] / batch_size)
    print(total_batch)
    for epoch in range(epochs):
        i = 0
        avg_cost = 0
        batches = batch_data(x_data,y_data,batch_size)
        for x_train, y_train in batches:
            _,l = sess.run([training_op, loss], feed_dict={X:x_train,Y:y_train})
            avg_cost = (avg_cost + l) / total_batch
            i += 1
            
        print("Epoch:", (epoch + 1), "cost=","{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={X:x_train,Y:y_train}))
    
