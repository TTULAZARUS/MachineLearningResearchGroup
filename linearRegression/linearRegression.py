import numpy as np

def calculateCost(features,weights,target):
  predictions = np.dot(features,weights)
  difference = np.subtract(predictions,target)
  difference = np.array(difference, dtype=np.float64)
  squaredDifference = np.square(difference)
  sumSquares = np.sum(squaredDifference)
  cost = np.multiply((1/len(target)),sumSquares)
  return cost

def gradientDescent(features,weights,target,learningRate,maxIterations,tolerance):
  costHistory = [calculateCost(features,weights,target)] 
  weightHistory = [weights] 
  weightModConst = (1/len(target))*learningRate
  iteration = 0
  cost = calculateCost(features,weights,target)
  while(cost > tolerance and iteration < maxIterations):
    predictions = np.dot(features,weights)
    difference = np.subtract(predictions,target)
    weightUpdate = np.dot(np.transpose(features),difference)
    weights = np.subtract(weights, np.multiply(weightModConst,weightUpdate))
    cost = calculateCost(features,weights,target)


    costHistory.append(cost)
    weightHistory.append(weights.tolist())
    iteration = iteration + 1
    print(iteration)

  return [weights, costHistory, weightHistory]


train = np.genfromtxt('bostonTrain.csv',delimiter=',',dtype=np.float64)
train = train[1:]
train = np.transpose(train)
train = train[1:]
train = np.transpose(train)

test = np.genfromtxt('bostonTest.csv',delimiter=',',dtype=np.float64)
test = np.transpose(test)
test = test[1:]
test = np.transpose(test)

featuresTrain = train[:,:-1]
targetTrain = train[:,-1]

featuresTest = test[:,:-1]
targetTest = test[:,-1]

learningRate = 0.000001
maxIterations = 1000000
tolerance = 0.001
weights = np.random.randn(len(featuresTrain[0,:]))

trainingResults = gradientDescent(featuresTrain,weights,targetTrain,learningRate,maxIterations,tolerance)

testResults = calculateCost(featuresTest,trainingResults[0],targetTest)
testResults = np.array([testResults])

np.savetxt('weights.csv', trainingResults[0], delimiter=',', fmt='%s')
np.savetxt('weightHistory.csv', trainingResults[2], delimiter=',', fmt='%s')
np.savetxt('costHistory.csv', trainingResults[1], delimiter=',', fmt='%s')
np.savetxt('testResults.csv', testResults, delimiter=',', fmt='%s')



