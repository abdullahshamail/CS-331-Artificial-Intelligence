import numpy as np
import random, statistics, math
import matplotlib.pyplot as plt

grid = np.zeros((8,8), np.int)
grid[:1,] = grid[-1:,] = grid[:,:1] = grid[:,-1:] = grid[4][-2] = grid[3][-2] = -1
grid[2][-2] = grid[3][-3] = grid[4][-3] = grid[5][-2] = grid[1:2,1:-1] = grid[-2:-1,1:-1] = grid[:,1:2] = 1

steps = 28
where = (1,1)
facing = 1
# 0 north, 1 east, 2 south, 3 west

def randomString():
	string = ''
	for x in range(1,(steps*2)+1):
		string += str(random.randint(0,1))
	fit = fitness(string)
	# print(fit)
	# if fit < 8:
		# return randomString()
	# else:
	return string


def initializePopulation(n):
	pop = []
	for x in range(1,n+1):
		pop.append(randomString())
	return pop

# 11 forward, 01 right, 10, left, 00 nothing from path

def extract(str):
	return (str[0:2],str[2:])

def allMoves(string):
	# print(string)
	moves = []
	while len(string) != 0:
		currentMove,remainingstr = extract(string)
		string = remainingstr
		moves.append(currentMove)
	return moves

def calcGrid(string):
	moves = allMoves(string)
	fitness = 1
	curr = where
	currentfacing = facing
	tempgrid = grid.copy()
	tempgrid[curr] = 0
	for x in moves:
		if x == '11':
			if currentfacing == 0 and (curr[0]-1 > -1) and tempgrid[curr[0]-1][curr[1]] != -1 :
				curr = (curr[0]-1,curr[1])
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 1 and  (curr[1]+1 < 8) and tempgrid[curr[0]][curr[1]+1] != -1:
				curr = (curr[0],curr[1]+1)
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 2 and  (curr[0]+1 < 8) and tempgrid[curr[0]+1][curr[1]] != -1:
				curr = (curr[0]+1,curr[1])
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 3 and (curr[1]-1 > -1) and tempgrid[curr[0]][curr[1]-1] != -1 :
				curr = (curr[0],curr[1]-1)
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
		elif x == '01':
			currentfacing = (currentfacing+1)%4
		elif x == '10':
			currentfacing = (currentfacing-1)%4 #turn left
	return tempgrid


def fitness(string):
	moves = allMoves(string)
	fitness = 1
	curr = where
	currentfacing = facing
	tempgrid = grid.copy()
	tempgrid[curr] = 0
	for x in moves:
		if x == '11':
			if currentfacing == 0 and (curr[0]-1 > -1) and tempgrid[curr[0]-1][curr[1]] != -1 :
				curr = (curr[0]-1,curr[1])
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 1 and  (curr[1]+1 < 8) and tempgrid[curr[0]][curr[1]+1] != -1:
				curr = (curr[0],curr[1]+1)
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 2 and  (curr[0]+1 < 8) and tempgrid[curr[0]+1][curr[1]] != -1:
				curr = (curr[0]+1,curr[1])
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
			elif currentfacing == 3 and (curr[1]-1 > -1) and tempgrid[curr[0]][curr[1]-1] != -1 :
				curr = (curr[0],curr[1]-1)
				fitness += tempgrid[curr]
				tempgrid[curr] = 0
		elif x == '01':
			currentfacing = (currentfacing+1)%4 #turn right
		elif x == '10':
			currentfacing = (currentfacing-1)%4 #turn left
	return fitness


def giveExpectedCount(outputs, popSize):
	arr = outputs.copy()
	tempArr = np.zeros(popSize, np.int)
	while np.sum(tempArr) < popSize:
		maxIndex = np.argmax(arr)
		maxVal = arr[maxIndex]
		if(np.sum(tempArr) + math.ceil(maxVal + 0.5) > popSize):
			maxIndex = np.argmax(arr)
			newVal = popSize - np.sum(tempArr)
			tempArr[maxIndex] = newVal
			break
		else:
			tempArr[maxIndex] = math.ceil(maxVal + 0.5)
			arr[maxIndex] = 0
	return tempArr




def expectedOutputs(fitnesses):
	popSize = len(fitnesses)
	total = sum(fitnesses)
	tot = 0
	probabilities = np.array([x/total for x in fitnesses])
	expectedOutputs = np.array([popSize*y for y in probabilities])
	expected = giveExpectedCount(expectedOutputs, popSize)
	return expected



def evaluatePopulation(population):
	fitnesses = []
	for x in population:
		fitnesses.append(fitness(x))
	return fitnesses


def selectParents(population, expected):
	pop = population.copy()
	newPop = []
	for i in range(len(expected)):
		for j in range(0, expected[i]):
			newPop.append(pop[i])

	return newPop

def mutate(string, popSize):
	ret = '' 
	length = len(string)
	for x in range(len(string)):
		prob = random.random()
		left = right = 0
		if (1/popSize) < (1/length):
			left = 1/popSize
			right = 1/length
		else:
			right = 1/popSize
			left = 1/length
		if(prob > left and prob < right):
			if string[x] == '1':
				ret += '0'
			else:
				ret += '1'
		else:
			ret += string[x]

	return ret


	

def crossover(population):
	pop = population.copy()
	size = len(population)
	newPop = []
	for i in range(0,size,2):
		first = pop[i]
		second = pop[i+1]
		crossAt = random.randint(0,2*steps)
		firstleft = first[:crossAt]
		firstright = first[crossAt:]
		secondleft = second[:crossAt]
		secondright = second[crossAt:]
		pc = random.random()
		if (pc > 0.6 and pc < 0.9): #crossover
			newOne = firstleft + secondright
			newTwo = secondleft + firstright
		else:
			newOne = first
			newTwo = second

		
		# mutOrNot1 = random.random()
		# mutOrNot2 = random.random()
		# if mutOrNot1 < 1/size:
		newOne = mutate(newOne, size)
		# if mutOrNot2 > 1/size:
		newTwo = mutate(newTwo, size)
		
		newPop.append(newOne)
		newPop.append(newTwo)
	return newPop

def selectNextGen(originalParents,offspring, fitnesses, offspringFitnesses):
	size = len(originalParents)
	nextGen = []
	nextGenFitnesses = []
	while (len(nextGen) != size):
		pMax = max(fitnesses)
		oMax = max(offspringFitnesses)
		if pMax > oMax:
			ind = np.argmax(fitnesses)
			nextGen.append(originalParents[ind])
			fitnesses[ind] = 0
			nextGenFitnesses.append(pMax)
		else:
			ind = np.argmax(offspringFitnesses)
			nextGen.append(offspring[ind])
			offspringFitnesses[ind] = 0
			nextGenFitnesses.append(oMax)

	return nextGen, nextGenFitnesses

def main():
	print("Initial Grid\n",grid)
	size = 20
	

	population = initializePopulation(size)
	fitnesses = evaluatePopulation(population)

	generation = 1
	gens = []
	fits = []
	gens.append(generation)
	fits.append(max(fitnesses))

	print("Generation",generation,"Max Fitness", max(fitnesses))
	while (max(fitnesses) != 20 and generation != 100000):
		expectedOutput = expectedOutputs(fitnesses)

		parents = selectParents(population, expectedOutput)
		originalParents = parents.copy()
		random.shuffle(parents)
		offspring = crossover(parents)

		offspringFitnesses = evaluatePopulation(offspring)
		population, fitnesses = selectNextGen(originalParents,offspring, fitnesses, offspringFitnesses)

		generation += 1
		gens.append(generation)
		fits.append(max(fitnesses))
		print("Generation",generation,"Max Fitness", max(fitnesses))
	# generation += 1
	# print("Generation",generation,"Max Fitness", max(fitnesses))
	# maxWhere = np.argmax(fitnesses)
	# maxString = population[maxWhere]
	# print(calcGrid(maxString))
	xmax = max(gens)
	ymax = max(fits)
	fig = plt.figure()
	plt.xlabel('Generations')
	plt.ylabel('Maximum Fitnesses')
	plt.title('Population Size '+str(size))
	plt.plot(gens,fits)
	plt.plot(xmax,ymax, 'bo')
	plt.annotate('Maximum at '+str(ymax), xy=(xmax, ymax), xytext=(xmax-20000, ymax-5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

	fig.savefig('graph'+str(size)+'.png')

main()