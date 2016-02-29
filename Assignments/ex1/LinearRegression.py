# Assigment 2
# To implement Linear Regression
#
# Zafarullah Mahmood
# zafarulllahmahmood@gmail.com
# Last Upated: 02/18/2016

import numpy as np 

def h(X, theta):
	''' outputs a vector of hypotheses given feature X and weights theta;
		assumes the bias term in X is already added.

		X: 		numpy matrix of m*n dimensions
		theta:	numpy matrix of n*1 dimensions
		returns	numpy matrix of m*1 dimensions
	'''
	assert X.shape[1] == theta.shape[0]

	return X.dot(theta)



def J(X, theta, y):
	''' Calculates the cost of regression hypothesis given feature X,
		labels y and weights theta

		X:		numpy matrix of m*n dimensions
		theta: 	numpy matrix of n*1 dimensions
		y :		numpy matrix of m*1 dimensions
		returns int ; the cost
	'''
	assert X.shape[0] == y.shape[0] and X.shape[1] == theta.shape[0]

	m = X.shape[0]
	return  (1.00/(2*m))*np.sum(np.square(h(X, theta) - y))


def gradient_descent(X, y, alpha, iters):
	''' returns the a numpy matrix of weights after running the
		Gradient Descent Algorithm iters times using learning rate
		alpha

		X:		numpy matrix of m*n dimensions
		y :		numpy matrix of m*1 dimensions 
		alpha: 	positive integer (the learning rate)
		iters:	positive integer (number of iterations for gradient descent)
		returns numpy matrix of n*1 dimensions
	'''
	assert X.shape[0] == y.shape[0]

	m = X.shape[0]
	n = X.shape[1]
	theta = np.matrix(np.zeros([n,1]))
	J_history = np.zeros(iters)

	for i in range(iters):
		theta = theta - (alpha/m)*(X.T).dot(h(X, theta) - y)
		J_history[i] = J(X, theta, y)

	return [theta, J_history]

#  Test Code
if __name__ == '__main__':

	# Load the data
	data = np.matrix(np.loadtxt('ex1data1.txt', delimiter = ','))
	X = data[:, 0]
	y = data[:, 1]
	m = X.shape[0]

	print 'Visualising Data...'

	import matplotlib.pyplot as plt
	plt.plot(X, y, 'rx')

	X = np.append(np.ones((m, 1)), X, axis = 1)

	theta, J_history = gradient_descent(X, y, 0.01, 1500)
	print 'Theta is found to be:'
	print theta

	result_x = np.matrix([[np.min(X)], [np.max(X)]])
	result_x1 = np.append(np.ones((2, 1)), result_x, axis = 1)
	result_y = h(result_x1, theta)
	plt.plot(result_x, result_y, linestyle='-')
	plt.show()

	num_iters = np.arange(1500)
	plt.plot(num_iters, J_history, linestyle='-')
	plt.show()






