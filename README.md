# Fully-connected-neural-network-on-MNIST
This neural network is written from scratch in Python for the MNIST dataset (no PyTorch). The neural network is trained on the Training Set using stochastic gradient descent.
In this programming assignment, I use a class to encapsulate the single layer neural network. For
simplicity, every member function modifies a parameter (Z, H, C, etc.). Then I defined a function
for training in which I used a for loop to iterate. In every iteration, I modify those parameters.
Actually, I achieved high accuracy (nearly 98%) when I set the number of hidden units as 400 and
number of iteration as 1000,000. However, it took me over half an hour to train. Therefore, at last I
set the number of hidden units as 200, the number of iteration as 200,000. I compared the
performance of the 3 activation functions and found that tanh > relu > sigmoid.
