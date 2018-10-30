import numpy as np
import random

def calculateCost(weights,features,targets):
  '''
  Calculates the cost for a given feature vector X and target vector Y based on current weights w. 
  '''

  n = len(targets)
  predictions = np.multiply(weights,features) #Calculate our predictions based on our current estimated weights
  cost = np.multiply((1/n),np.sum(np.square(np.subtract(predictions,targets))))
  return cost



def gradientDescent(weights, features, targets, learningRate, maxIterations,tolerance):

  n = len(targets)
  gradientScalar = np.multiply((-2/n),learningRate)
  cost = calculateCost(weights, features, targets)
  costHistory = [cost]
  weightHistory = [weights]

  for iteration in range(maxIterations):
    predictions = np.multiply(weights,features) #Update predictions based on current weights

    weightModifier = gradientScalar*np.sum(np.multiply(np.subtract(predictions,targets),features)) 
    weights = np.subtract(weights,weightModifier)

    cost = calculateCost(weights,features,targets)

    weightHistory.append(weights)
    costHistory.append(cost)

    if (cost < tolerance):
      break
  return [weights, weightHistory, costHistory]



data = np.genfromtxt('train.csv', delimiter=',') 
features = data[0] #Set features to be our first column vector.
targets = data[1] #Set targets to be our second(and last) column vector.

learningRate = 0.001 #This is the scalar that adjusts our gradient step. Think of step size. 
maxIterations = 1000 #The maximum number of iterations we will allow before stopping our algorithm.
tolerance = 0.01 #If our cost function is less than this number, we will stop our algorithm. 
weights = random.randint(-50,50) #Initialize random values for our weights. 

trainingResults = gradientDescent(weights, features, targets, learningRate, maxIterations, tolerance)

np.savetxt('weights.csv', [trainingResults[0]], delimiter=',', fmt='%s')
np.savetxt('weightHistory.csv', trainingResults[1], delimiter=',', fmt='%s')
np.savetxt('costHistory.csv', trainingResults[2], delimiter=',', fmt='%s')

