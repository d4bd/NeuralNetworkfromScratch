/*
Created: 24/10/21
Author: Space
Object: Implementation of the dense layer class for the output layer.
*/
#include<iostream>

#include "DenseLayerOutput.hh"
#include "DenseLayer.hh"
#include "Matrix.hh"
#include "ActivationFunction.hh"

DenseLayerOutput::DenseLayerOutput()
    : DenseLayer() {}

DenseLayerOutput::DenseLayerOutput(unsigned int inputNum, unsigned int neurons)
    : DenseLayer(inputNum , neurons) {
    std::cout << "The layer just constructed was on output layer\n";
    }

DenseLayerOutput::DenseLayerOutput(unsigned int inputNum, unsigned int neurons, Matrix weights, Vector biases)
    : DenseLayer(inputNum , neurons, weights, biases) {
    std::cout << "The layer just constructed was on output layer\n\n";
    }

void DenseLayerOutput::errorCalculation(const Matrix & correctOutput)
{
    m_error = (layerActivation() - correctOutput).cwiseProduct(DerivativeSigmoidActivationFunction(layerWeightedInput()));
}