# NeuralNetworkfromScratch
Implementation of a neural network from scratch in c++ for the classification of hand written digits.
The program is implemented through a series of classes and functions:
- TypedefinitionsAndGlobals: stores all user type definitions and global variables
- Handler: implements the handler, its function is to allow to save the progress made on the training of the network befor shutting down. It operates by interrupting CTRL + C
- Matrix: contains some generic functions related to matrix creation and manipulation, for example the creation of a random matrix of specified dimensions
- ActivationFunction: implements the functions for the calculation of the activation functions of each layer of neurons, i.e. on matrices
- DenseLayer: is the main class of the program. Instead of implementing each neuron singularly it was decided to implement the layers. This class, other than storing the values of the weights and biases of the layer in form of a matrices (since for the training stocastic gradient descend is used), handles the forward and backward propagation at leyer level and other things
- DenseLayerOutput: is a class derived from the DenseLayer class. In this class is implemented the method for the error calculation7
- NeuralNetwork: the class describing the neural network, substantially is a list of neuron layers plus an output layer. It handles the creation of the network both new or form an existing save and can save the network. It also handles training, testing and the data collection. For the evaluation of the performances data as time for training and percentage of correct guessing in testing are stored
- LoadSaveNetwork: functions to save and load the neural nerwork
- ReadMinst: are the functions use for reading the MNIST database whihc is used for the training and testing. In the current inmplementation to load the database it's necessary to provide the uncompressed files inside of a directory denominated "MNIST" in the same directory in which the compiled program it's executed

Other observations reguarding the project:
- The geometry used was 784,100,10
- Sigmod neurons were used
- For the training is used the gradient descend algorithm
- The cost function used is the quadratic cost function
- The library used for the algebric calculations is Eigen
- For the theory behind the project reference was made to the book by Michael Nielsen: http://neuralnetworksanddeeplearning.com/
