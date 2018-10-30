import numpy as np
import random

def calculateCost(weights,features,targets):
  '''
  Calculates the cost for a given feature vector X and target vector Y based on current weights w. 
  '''

  n = len(targets)
  predictions = np.dot(features,weights) #Calculate our predictions based on our current estimated weights
  cost = np.multiply((1/n),np.sum(np.square(np.subtract(predictions,targets))))
  return cost



def gradientDescent(weights, features, targets, learningRate, maxIterations,tolerance):

  n = len(targets)
  gradientScalar = np.multiply((2/n),learningRate)
  cost = calculateCost(weights, features, targets)
  costHistory = [cost]
  weightHistory = weights.tolist()

  for iteration in range(maxIterations):
    predictions = np.dot(features,weights) #Update predictions based on current weights

    #weightModifier = gradientScalar*np.dot(features,np.subtract(predictions,targets)) 
    weightModifier = gradientScalar*(features.T.dot(np.subtract(predictions,targets)))
    weights = np.subtract(weights,weightModifier)

    cost = calculateCost(weights,features,targets)

    weightHistory.append(weights.tolist())
    costHistory.append(cost)

    if (cost < tolerance):
      break
  return [weights, weightHistory, costHistory]



data = np.genfromtxt('bostonTrain.csv', delimiter=',') 
data = data[1:] 
data = np.transpose(data) 
data = data[1:]
data = np.transpose(data)


features = data[:,:-1] #Set features 
targets = data[:,-1] #Set targets 

learningRate = 0.001 #This is the scalar that adjusts our gradient step. Think of step size. 
maxIterations = 1000 #The maximum number of iterations we will allow before stopping our algorithm.
tolerance = 0.01 #If our cost function is less than this number, we will stop our algorithm. 
weights = np.random.randn(len(features[0,:]),1) #Initialize random values for our weights. 

trainingResults = gradientDescent(weights, features, targets, learningRate, maxIterations, tolerance)

np.savetxt('weights.csv', [trainingResults[0]], delimiter=',', fmt='%s')
np.savetxt('weightHistory.csv', trainingResults[1], delimiter=',', fmt='%s')
np.savetxt('costHistory.csv', trainingResults[2], delimiter=',', fmt='%s')

