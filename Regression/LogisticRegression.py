import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

inputFile = open("redwineData.txt", 'r')
dataDescripter = []

while True:
	inputLine = inputFile.readline()
	if not inputLine : break
	inputLine = inputLine.split('\t')
	inputs=[]
	for i in range(len(inputLine)):
		inputs.append(float(inputLine[i]))
	dataDescripter.append(inputs)

#for i in dataDescripter:
#	print i
#Training Data
train=[[] for i in range(len(dataDescripter[0])-1)]
t_Y=[]

for i in range(len(dataDescripter)):
	for j in range(len(dataDescripter[i])-1):
		train[i].append(dataDescripter[i][j])
	t_Y.append(dataDescripter[i][11])

train_Y=np.asarray(t_Y)
train_X = np.asarray(train)
#print train
print len(dataDescripter), len(train_X[0]), len(train_Y)
n_attribute = train_X.shape[0]

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [11]) 
y = tf.placeholder("float", [1]) # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([1599, 1]))
b = tf.Variable(tf.zeros([11]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(1599)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train_X[i], train_Y[i]
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    #correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
