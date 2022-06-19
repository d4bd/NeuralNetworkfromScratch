/*
Created: 19/10/21
Author: Space
Object: Implementatin of the dense layer class.
*/

#include <iostream>

#include "Matrix.hh"
#include "ActivationFunction.hh"
#include "DenseLayer.hh"

DenseLayer::DenseLayer() {}

DenseLayer::DenseLayer(unsigned int inputNum, unsigned int neurons)
	: m_inputNum(inputNum), m_neurons(neurons) {
	std::cout << "Constructing dense layer: inputNum = " << inputNum << " , neurons = " << neurons << "\n";
	if (neurons < 1 || inputNum < 1)
	{
		std::cerr << "Wrongly defined dense layer, has no neurons inside or no input" << std::endl;
		exit(2);
	}

	//In the beginning the weights associated to each connection are random (extracted from a normal distribution(0,1))
	m_weights = RandomMatrix(neurons, inputNum);
	m_biases = RandomVector(neurons);
}

DenseLayer::DenseLayer(unsigned int inputNum, unsigned int neurons, Matrix weights, Vector biases)
	: m_inputNum(inputNum), m_neurons(neurons), m_weights(weights), m_biases(biases) {
		std::cout << "Constructing dense layer: inputNum = " << inputNum << " , neurons = " << neurons << "\n";
}

void DenseLayer::propagateForward(const Matrix & input)
{
	m_weightedInput = (m_weights * input).colwise() + m_biases;
	m_activation = SigmoidActivationFunction(m_weightedInput);
}

Matrix DenseLayer::propagateBackward(const Matrix & previousLayerWeightedInputs)
{
	return (m_weights.transpose()*m_error).cwiseProduct(DerivativeSigmoidActivationFunction(previousLayerWeightedInputs));
}

void DenseLayer::derivativeCostFunctionWeights(const Matrix & previousLayerActivation)
{
	m_dWeights = (m_error*(previousLayerActivation.transpose()))/m_error.cols();
}

void DenseLayer::derivativeCostFunctionBiases()
{
	m_dBiases = m_error.rowwise().mean();
}

void DenseLayer::SGD(const float & learnignRate, const Matrix & previousLayerActivation)
{
	//Calculates the mean derivative with respect to the weights on the sample batch
	derivativeCostFunctionWeights(previousLayerActivation);
	m_weights -= learnignRate*m_dWeights;

	//Calculates the mean derivative with respect to the biases on the sample batch
	derivativeCostFunctionBiases();
	m_biases -= learnignRate*m_dBiases;
}
