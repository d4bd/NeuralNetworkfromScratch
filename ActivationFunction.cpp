/*
Created: 22/10/21
Author: Space
Object: Implementation of the activatin functions of the neurons
*/

#include <algorithm>
#include <math.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

#include "Matrix.hh"
#include "ActivationFunction.hh"

float Sigmoid(float value)
{
    return 1/(1+expf(-value));
}

Matrix SigmoidActivationFunction(Matrix & output)
{
    return output.unaryExpr(&Sigmoid);
}

float DerivativeSigmoid(float value)
{
    return expf(value)/(powf((expf(value) + 1),2));
}

Matrix DerivativeSigmoidActivationFunction(const Matrix & output)
{
    return output.unaryExpr(&DerivativeSigmoid);
}