import tensorflow as tf
import matplotlib.pyplot as plt
import collections
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
trainNum = 1500
testNum = totalNum-trainNum
attributeNum = len(dataDescripter[0])-1
#for i in dataDescripter:
#	print i
#Training Data
trainX=[[] for i in range(trainNum)]
trainY=[]
testX = [[] for i in range(testNum)]
testY=[]

for i in range(totalNum):
	for j in range(len(dataDescripter[i])-1):
		if i< trainNum:
            		trainX[i].append(dataDescripter[i][j])
        	else:
            		testX[i-trainNum].append(dataDescripter[i][j])
	if i< trainNum:
       		trainY.append(dataDescripter[i][attributeNum])
    	else:
        	testY.append(dataDescripter[i][attributeNum])

Xtr = np.asarray(trainX)
Ytr = np.asarray(trainY)
Xte = np.asarray(testX)
Yte = np.asarray(testY)

xtr = tf.placeholder("float")
xte = tf.placeholder("float")

L1 = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
	

accuracy = 0.

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        L1matrix = sess.run(L1, feed_dict={xtr:Xtr, xte:Xte[i]})
	indices = []
	for j in range(15):
		index = np.argmin(L1matrix, 0)
		L1matrix = np.delete(L1matrix, index)
		indices.append(Ytr[index])
	counter = collections.Counter(indices)
	nn_index = float(counter.most_common()[0][0])
        print "Test", i, "Prediction:",nn_index, "True Class:",Yte[i]

        if nn_index==Yte[i]:
            accuracy += 1./len(Xte)

    print "Done"
    print "Accuracy:", accuracy

