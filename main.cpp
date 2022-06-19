#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include "Handler.hh"
#include "Matrix.hh"
#include "ActivationFunction.hh"
#include "DenseLayer.hh"
#include "NeuralNetwork.hh"
#include "DenseLayerOutput.hh"
#include "ReadMinst.hh"

int main()
{
	//MNIST CODE	
	NeuralNetwork net({784,30,10},1.0);
	
	std::vector<Vector> inputs;
	std::vector<Vector> correctOutputs;
	std::vector<Vector> checkInputs;
	std::vector<Vector> checkCorrectOutpust;

	std::string trainingData = "./MNIST/train-images.idx3-ubyte";
	std::string labelTrainingData = "./MNIST/train-labels.idx1-ubyte";
	std::string checkData = "./MNIST/t10k-images.idx3-ubyte";
	std::string labelCheckData = "./MNIST/t10k-labels.idx1-ubyte";
	
	read_mnist_cv(inputs, correctOutputs, trainingData,labelTrainingData);

	read_mnist_cv(checkInputs, checkCorrectOutpust, checkData, labelCheckData);

    auto start = std::chrono::steady_clock::now();
	net.train(inputs, correctOutputs, 10, 30, checkInputs, checkCorrectOutpust);
    auto endTime = std::chrono::steady_clock::now();

    auto diff = endTime - start;
    auto diff_sec = std::chrono::duration_cast<std::chrono::seconds>(diff);
    std::cout << "Time: " << diff_sec.count() << " seconds"<< std::endl;
	
    return 0;
}