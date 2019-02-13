import numpy as np
import h5py
import time
import copy
from random import randint

class SingleLayerNN:
	def __init__(self, input_size, output_size, hidden_units, activation):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_units = hidden_units
		self.activation = activation
		self.learningRate = learningRate
		self.itr = itr

	def get_W(self):
		self.W = np.random.randn(self.hidden_units, self.input_size) / np.sqrt(self.input_size)		

	def get_bias1(self):
		self.bias1 = np.random.randn(self.hidden_units, 1)

	def get_Z(self, x): #get the pre hidden
		self.Z = np.matmul(self.W, x) + self.bias1

	def get_H(self): #get the hidden layer
		self.H = self.Z
		if self.activation == 'relu':
			for i in range(len(self.Z)):
				if self.Z[i] < 0:
					self.H[i] = 0
		elif self.activation == 'tanh':
			self.H = np.tanh(self.H)
		elif self.activation == 'sigmoid':
			for i in range(len(self.Z)):
				element = self.Z[i]
				self.H[i] = np.exp(element)/(1+np.exp(element))

	def get_C(self):
		self.C = np.random.randn(self.output_size, self.hidden_units)

	def get_bias2(self):
		self.bias2 = np.random.randn(self.output_size, 1)

	def get_U(self): #get the pre softmax
		self.U = np.matmul(self.C, self.H) + self.bias2
	
	def get_f(self):#softmax
		total = 0
		for entry in self.U:
			total += np.exp(entry)
		processed = []
		for entry in self.U:
			processed.append(np.exp(entry)/total)
		self.f = np.array(processed).reshape(self.output_size, 1)
		return self.f

	def indicator_vec_generate(self, y):
		List = []
		for k in range(10):
			if y == k:
				List.append(1)
			else:
				List.append(0)
		return np.array(List).reshape(10,1)

	def partial_rho_partial_U(self, y):
		self.rho_partial_U = self.f - self.indicator_vec_generate(y)

	def get_delta(self, y):
		self.delta = np.matmul(self.C.transpose(), self.rho_partial_U)

	def sigma_prime(self):
		self.sigma_prime_Z = self.Z
		if self.activation == 'relu':
			for i in range (len(self.Z)):
				if self.Z[i] <= 0:
					self.sigma_prime_Z[i] = 0
				else:
					self.sigma_prime_Z[i] = 1
		elif self.activation == 'tanh':
			self.sigma_prime_Z = 1 - np.tanh(self.Z)**2
		elif self.activation == 'sigmoid':
			for i in range(len(self.Z)):
				element = self.Z[i]
				sigma = np.exp(element)/(1+np.exp(element))
				self.sigma_prime_Z[i] = sigma*(1-sigma)


input_size = 784
output_size = 10
hidden_units = 200
activation = 'tanh'
learningRate = 0.0036
itr = 200000
def train(theta, x_train, y_train):
	index_set = np.random.choice(60000,itr,replace = True) #the size of x_train is 60000
	for l in range(itr):
		index = index_set[l]
		x = x_train[index]
		x = x.reshape(input_size,1)
		y = y_train[index]
		if l == 0:
			theta.get_W()
			theta.get_bias1()
		theta.get_Z(x)
		theta.get_H()
		if l == 0:
			theta.get_C()
			theta.get_bias2()
		theta.get_U()
		theta.get_f()
		theta.partial_rho_partial_U(y)
		theta.get_delta(y)
		theta.sigma_prime()
		theta.W -= learningRate * np.matmul((theta.delta * theta.sigma_prime_Z),x.transpose())
		theta.bias1 -= learningRate * (theta.delta * theta.sigma_prime_Z)
		theta.C -= learningRate * np.matmul(theta.rho_partial_U, theta.H.transpose())
		theta.bias2 -= learningRate * theta.rho_partial_U
		print('Iteration ' + str(l) + '/' + str(itr))



def softmax(x):
	total = 0
	for i in x:
		total += np.exp(i)
	processed_x = []
	for i in x:
		processed_x.append(np.exp(i)/total)
	return np.array(processed_x)

def forward(x, model, activation):
	Z = np.matmul(model.W, x) + model.bias1
	if activation == 'relu':
		H = Z
		for i in range(len(Z)):
			if Z[i] <= 0:
				H[i] = 0
	elif activation == 'tanh':
		H = np.tanh(Z)
	elif activation == 'sigmoid':
		H = Z
		for i in range(len(Z)):
			element = Z[i]
			H[i] = np.exp(element)/(1+np.exp(element))
	U = np.matmul(model.C, H) + model.bias2
	f = softmax(U)
	return f

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )

MNIST_data.close()

theta = SingleLayerNN(input_size = input_size, output_size = output_size, 
	hidden_units = hidden_units, activation = activation)
train(theta, x_train, y_train)

total_correct = 0
for n in range(len(x_test)):
	y = y_test[n]
	x = x_test[n][:].reshape(input_size, 1)
	p = forward(x, theta, activation)
	prediction = np.argmax(p)
	if (prediction == y):
		total_correct += 1
accuracy = total_correct/np.float(len(x_test) )
print(accuracy)
