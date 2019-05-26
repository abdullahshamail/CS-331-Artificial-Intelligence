import numpy as np
import os, sys
import cv2
import time
np.seterr(all='ignore')

numImagesTrain = 60000

if len(sys.argv) != 6:
	print("Usage: ", sys.argv[0], "train/test data labels lr weightsFile")
	os._exit(0)

def takeInput():
	what = sys.argv[1]
	data = sys.argv[2]
	labels = sys.argv[3]
	lr = float(sys.argv[4])
	weightsFile = sys.argv[5]

	return what, data, labels, lr, weightsFile

def normalise(image):
    m = np.mean(image)
    s = np.std(image)

    image = (image - m) / s
    return image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def readFile(file):
    global numImagesTrain
    # print("Opening file")
    with open(file, 'r') as f:
        allImages = []
        count = 0
        image = ""
        for line in f:
            if(count % 44 == 0 and count != 0):
                count = 0
                im = image.split()
                im = list(map(int, im))
                allImages.append(im)
                image = ""
            line = line.strip('[')
            line = line.replace("]", "")
            image = image + line
            count += 1
    allImages = np.array(allImages)
    return allImages

def readLabels(file):
    labels = []
    with open(file, 'r') as f:
        labs = f.readlines()
        for lab in labs:
            lab = lab.strip('\n')
            labels.append(int(lab))
        
        labels = np.array(labels)
    return labels


def train(epochs, images, labels, lr):
	W_h = np.random.normal(size=(784, 30))
	W_o = np.random.normal(size=(30, 10))
	weights_ = [W_h, W_o]
	acc = 0
	for epoch in range(epochs):
		acc = 0
		for i in range(len(images)):
		    sample_input = np.matrix(images[i])
		    sample_input = normalise(sample_input)

		    activations = np.dot(sample_input, weights_[0])
		    activations = sigmoid(activations)
		    
		    output = np.dot(activations, weights_[1])
		    output = sigmoid(output)
		    

		    t = np.argmax(output)

		    if(t == labels[i]):
		    	acc += 1

		    errors = np.zeros((1,10))
		    errors[0, labels[i]] = 1
		    
		    err = output - errors
		    err = np.transpose(err)

		    #backprop
		    dhm = np.multiply(activations, 1 - activations)
		    dh2m = np.dot(weights_[1], err)
		    dh2m = np.transpose(dh2m)
		    dh = np.multiply(dhm, dh2m)
		    
		    weights_[0] = weights_[0] - lr * np.dot(sample_input.T, dh)
		    weights_[1] = weights_[1] - lr * np.dot(activations.T, err.T)
		print ("After Epoch number ",epoch + 1, ": ",acc,"/",len(images)+1," images correctly classified")
		acc = ((acc * 100) / len(images)+1)
		print("Accuracy: ",acc,"%")
		print("Error: ", 100 - acc, "%")
	        
	filee = open("hiddenWeights.txt", "w")
	np.savetxt(filee, weights_[0])
	filee.close()
	filee = open("outputWeights.txt", "w")
	np.savetxt(filee, weights_[1])
	filee.close()

	with open("weights.txt", "w") as f:
	    arr = "hiddenWeights.txt, outputWeights.txt"
	    f.write(arr)

	return acc


def test(images, labels, weightsFile):
	hiddenWeights = None
	outputWeights = None
	with open("weights.txt", "r") as f:
	    arr = f.readline()
	    files = arr.split(", ")
	    hiddenWeights = np.loadtxt(files[0])
	    outputWeights = np.loadtxt(files[1])

	acc = 0
	for i in range(len(images)):
		input_layer = np.matrix(images[i])
		input_layer = normalise(input_layer)

		hidden_layer = np.dot(input_layer, hiddenWeights)
		hidden_layer = sigmoid(hidden_layer)

		output_layer = np.dot(hidden_layer, outputWeights)
		output_layer = sigmoid(output_layer)

		target = np.argmax(output_layer)

		if target == labels[i]:
			acc +=1

	print("On test data")
	acc = ((acc * 100) / len(images)+1)
	print("Accuracy: ",acc,"%")
	print("Error: ", 100 - acc, "%")

def main():
	option, data, labelFile, lr, weightsFile = takeInput()
	if(option == "train"):
		t1 = time.time()
		images = readFile(data)
		labels = readLabels(labelFile)
		# print("im", images[0:1])
		# print("lab", labels[0:10])
		print("Going to train")
		acc = train(2, images, labels, lr)
		t2 = time.time()

		print(f"Time taken {t2 - t1}")

		with open("lrAndTime.txt", "a+") as f:
			f.write(str(lr) + "," + str(t2 - t1) + "," + str(acc) + "\n")
	if(option == "test"):
		images = readFile(data)
		labels = readLabels(labelFile)
		test(images, labels, weightsFile)



main()