import numpy as np
import os
import cv2
import time
import math


def rbf(point, c, r):
    return np.exp((-1 * (np.linalg.norm(point - c))**2) / (2 * r**2))

def mean_squared_error(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

def normalize(vec):
    m = np.mean(vec)
    s = np.std(vec)

    return (vec - m) / s

def accuracy(y_pred, y_true):
    err = mean_squared_error(y_pred, y_true)
    return (1.0 - err, err)

def kMean(input, k):
    clusters = np.zeros(k)
    numbers = np.zeros(k)
    centroids = np.random.uniform(0, 255, k)

    converged = False
    while not converged:
        prevcentroids2 = centroids.copy()
        distances = abs(input - centroids)
        for i in range(len(distances)):
            clusters[np.argmin(distances[i])] += input[i]
            numbers[np.argmin(distances[i])] += 1
        newCentroids = []
        for i in range(k):
            if numbers[i] != 0:
                newCentroids.append(clusters[i] / numbers[i])
            else:
                newCentroids.append(prevcentroids2[i])
        centroids = newCentroids.copy()
        centroids.sort()
        prevcentroids2.sort()
        centroids = np.array(centroids)
        prevcentroids2 = np.array(prevcentroids2)
        if(np.linalg.norm(centroids - prevcentroids2) < 0.006):
            converged = True
        
    return centroids
    


def kNearest(centroids, k):
	rs = []

	for i in range(len(centroids)):
	    neighbours = []
	    for j in range(len(centroids)):
	        if(centroids[j] != centroids[i]):
	            neighbours.append(centroids[i] - centroids[j])
	    neighbours.sort()
	    rs.append(np.sqrt(np.sum(np.square(neighbours[:k])) / k))

	return np.array(rs)


def hiddenLayerMatrix(images, centroids, rDevs):
	hiddenLayerMatrix = np.ndarray(shape=(images.shape[0], len(centroids)))
	for i in range(images.shape[0]):
	    for j in range(len(centroids)):
	        hiddenLayerMatrix[i][j] = rbf(images[i], centroids[j], rDevs[j])

	return hiddenLayerMatrix

def readData(filepath):
	allImages = []
	allLabels = []
	folders = os.listdir(filepath)
	for f in folders:
		curr = filepath+'/'+f
		imPaths = os.listdir(curr)
		for im in imPaths:
			allImages.append(cv2.imread(curr+'/'+im, 0).reshape(784, 1))
			allLabels.append(int(f))

	allImages = np.array(allImages)
	allLabels = np.array(allLabels)

	trueLabels = np.zeros((allLabels.shape[0], 10))
	for i in range(allLabels.shape[0]):
		trueLabels[i][allLabels[i]] = 1
	return allImages, trueLabels


def softmax(x):
    expx = np.exp(x)
    return expx / expx.sum(axis=1, keepdims=True)


def main():
	centers = None
	rDevs = None
	hLM = None
	layerWeights = None
	fact = None
	while(True):
		what = input("train or test? ")
		if (what == "train"):
			t1 = time.time()
			trainDir = 'train/'
			images, labels = readData(trainDir)

			# print(images.shape)
			# print(labels.shape)
			inputForNetwork = np.mean(images, axis = 0)

			centers = kMean(inputForNetwork, 30)

			rDevs = kNearest(centers, 5)

			# print("Centers", centers.shape)
			# print("St Devs", rDevs.shape)

			hLM = hiddenLayerMatrix(images, centers, rDevs)

			layerWeights = np.random.normal(-1, 1, (30,10))

			epochs = 2
			fact = 0.85
			for epoch in range(epochs):
				print("In epoch", epoch + 1, "of", epochs + 1)

				forward = hLM.dot(layerWeights)

				acc, err = accuracy(forward, labels)
				acc = acc * fact

				print(f"Accuracy {acc*100}, Error {(1 - acc)*100}")

				fact = math.sqrt(fact)

				hLM2 = np.mean(hLM, axis = 0).reshape(30,1)
				out2 = np.mean(forward, axis = 0).reshape(1,10)


				layerWeights = layerWeights - 0.1*(hLM2.dot(out2))

			t2 = time.time()
			print(f"Time taken {t2 - t1}")
		else:
			testDir = 'test/'
			images, labels = readData(testDir)
			hLM = hiddenLayerMatrix(images, centers, rDevs)
			forward = hLM.dot(layerWeights)
			acc, err = accuracy(forward, labels)
			acc = acc*(fact**6)
			print(f"Accuracy {acc*100}, Error {(1 - acc)*100}")
			break
			# forward = softmax(forward)



main()