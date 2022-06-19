/*
Created: 22/10/21
Author: Space
Object: Declaration of the activatin functions of the neurons
*/

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.hh"

float Sigmoid(float value);
Matrix SigmoidActivationFunction(Matrix & output);

float DerivativeSigmoid(float value);
Matrix DerivativeSigmoidActivationFunction(const Matrix & output);

#endif