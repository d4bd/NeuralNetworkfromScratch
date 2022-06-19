/*
Created: 24/10/21
Author: Space
Object: Header file for the dense layer class for the output layer.
        The reason behind this is the need of a different function to calculate the error associated to the layer, and in some cases the use of a different activation function
*/

#ifndef DENSELAYEROUTPUT_H
#define DENSELAYEROUTPUT_H

#include "DenseLayer.hh"

class DenseLayerOutput : public DenseLayer     
{
public:
    DenseLayerOutput();
    DenseLayerOutput(unsigned int inputNum, unsigned int neurons);
    DenseLayerOutput(unsigned int inputNum, unsigned int neurons, Matrix weights, Vector biases);

    
    //Error calculation
    //Function to calculate the error Matrix associated to the output layer
    void errorCalculation(const Matrix & correctOutput);
};

#endif