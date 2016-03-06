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


totalNum = len(dataDescripter)
trainNum = 1000
testNum = totalNum-trainNum
attributeNum = len(dataDescripter[0])-1
#for i in dataDescripter:
#	print i
#Training Data
trainX=[[] for i in range(len(dataDescripter[0])-1)]
trainY=[]
testX = [[] for i in range(len(dataDescripter[0])-1)]
testY=[]

for i in range(totalNum):
	for j in range(len(dataDescripter[i])-1):
		if i>= trainNum:
            trainX[i].append(dataDescripter[i][j])
        else:
            testX[i-trainNum].append(dataDescripter[i][j])
	if i>= trainNum:
        train_Y.append(dataDescripter[i][attributeNum])
    else:
        test_Y.append(dataDescripter[i][attributeNum])

Xtr = np.asarray(trainX)
Ytr = np.asarray(trainY)
Xte = np.asarray(testX)
Yte = np.asarray(testY)

xtr = tf.placeholder("float")
xte = tf.placeholder("float")

L1 = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr:Xtr, xte:Xte[i]})
        print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i])

        if np.argmax(Ytr[nn_index])==np.argmax(Yte[i]):
            accuracy += 1./len(Xte)

    print "Done"
    print "Accuracy:", accuracy

