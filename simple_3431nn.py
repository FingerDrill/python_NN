import numpy as np

# Convert scientific notation to float
np.set_printoptions(suppress=True)
np.set_printoptions(precision=25)


# Epoch
epoch = 50000
print("epoch :", epoch, '\n')


# Learning rate
alpha = 0.2
print("Learning rate :", alpha, '\n')


### For reproduction
np.random.seed(421)


### Activation function : sigmoid
### 'deriv=True' to use derivative
### when calculate delta for back-propagation
def sigmoid(x, deriv=False) :
	if deriv==True :
		return x*(1-x)
	return 1/(1+np.exp(-x))


### structure : 3-4-3-1 neural network
### 4 rows, 3 variables for input
input = np.array([ [0,0,1],
				   [0,1,1],
				   [1,0,1],
				   [1,1,1] ])
target = np.array([[0,0,1,1]]).T
print('input :\n', input)


### Weight initialize
weight_hidden1 = np.random.randn(3,4)
weight_hidden2 = np.random.randn(4,3)
weight_output = np.random.randn(3,1)


### Training
for i in range(epoch) :

	### Feed-forward
	hidden1 = sigmoid(np.dot(input, weight_hidden1))
	hidden2 = sigmoid(np.dot(hidden1, weight_hidden2))
	output = sigmoid(np.dot(hidden2, weight_output))
		
		
	### Calculate delta
	delta_output = output-target
	delta_hidden2 = delta_output.dot(weight_output.T)*sigmoid(hidden2, deriv=True)
	delta_hidden1 = delta_hidden2.dot(weight_hidden2.T)*sigmoid(hidden1, deriv=True)

		
	### Back-propagation
	weight_output -= alpha*np.dot(hidden2.T, delta_output)
	weight_hidden2 -= alpha*np.dot(hidden1.T, delta_hidden2)
	weight_hidden1 -= alpha*np.dot(input.T, delta_hidden1)
	
	
	### Loss function
	loss = - np.dot(target.T, np.log(output)) - np.dot((1-target).T, 1-np.log(output))

	
	### Print status
	if i % 5000 == 0 :
		print('\n Total error at', i, 'th loop :', loss)
		print('Output : \n', output)

	


				   


