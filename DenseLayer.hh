/*
Created: 19/10/21
Author: Space
Object: Header file for the dense layer class. In this progect we will work with dense or fully connected layers of neurons which means that each neuron of a layer is connected to all the neurons of the preceding layer
*/

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Matrix.hh"

class DenseLayer
{
public:
    DenseLayer();
    DenseLayer(unsigned int inputNum, unsigned int neurons);
    DenseLayer(unsigned int inputNum, unsigned int neurons, Matrix weights, Vector biases);
    
    //Selector methods
    unsigned int inputsNum() const {return m_inputNum;};
    unsigned int neuronsNum() const {return m_neurons;};
    Matrix layerWeights() const {return m_weights;};
    Vector layerBiases() const {return m_biases;};
    Matrix layerWeightedInput() const {return m_weightedInput;};
    Matrix layerActivation() const {return m_activation;};

    //Modifier methods
    void setError(const Matrix & calculatedError) {m_error = calculatedError;};     //Method to associate error to the layer 
    void propagateForward(const Matrix & input);                                    //Returns the output of the hidden layer, i.e. uses the ReLu activation function
    Matrix propagateBackward(const Matrix & previousLayerWeightedInputs);           //Calculates the error of the previous layer based on the error of this layer
    void derivativeCostFunctionWeights(const Matrix & previousLayerActivation);
    void derivativeCostFunctionBiases();
    void SGD(const float & learnignRate, const Matrix & previousLayerActivation);
  
private:
    unsigned int m_inputNum;    //Number of inputs for each neuron of the layer, i.e. number of neurons of the preceding layer
    unsigned int m_neurons;     //Number of neurons of the layer

    Matrix m_weights;           //Weights associated to each connection of each neuron (is a matrix m_neurons x m_inputNum)
    Vector m_biases;            //Biases associated to each neuron (is a vector of dimension m_neurons)

    Matrix m_weightedInput;     //Matrix to store the weighted input of the Layer (is a matrix m_neurons x m_inputNum)
    Matrix m_activation;        //Matrix to store the activation of the Layer (is a matrix m_neurons x m_inputNum obtained by applying the activation function to m_weightedInput)

    Matrix m_dWeights;          //Stores the derivative of the cost function with respect to the weights
    Vector m_dBiases;           //Stores the derivative of the cost function with respect to the biases

protected:
    Matrix m_error;             //Error associated to the layer
};

#endif