/*
Created: 19/10/21
Author: Space
Object: Implementation of a series of function centered around matrix manipulation
*/

#include <iostream>
#include <sstream>

#include "Matrix.hh"

//Random generator -> Generates numbers according to a normal distribution
std::random_device rd; //rd è il seed per il generatore
std::mt19937 generator(rd());
std::normal_distribution<float> distribution(0.0,1.0);

inline float addNormal(const float & value)
{
    return value + distribution(generator);
}

Matrix RandomMatrix(const unsigned int & rowNum, const unsigned int & coloumnNum)
{
    Matrix m = Matrix::Zero(rowNum,coloumnNum).unaryExpr(&addNormal);
    return m;
}

Vector RandomVector(const unsigned int & dimension)
{
    Vector vec = Vector::Zero(dimension).unaryExpr(&addNormal);
    return vec;
}

Vector correctOutputVector(const char & correctOutputLabel)
{
    //Return a vector with the label position as 1 and all other elements as zeros
    Vector empty = Vector::Zero(10);
    empty(int(correctOutputLabel)) = 1;
    return empty;
}