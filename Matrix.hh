/*
Created: 19/10/21
Author: Space
Object: Header file for a series of function centered around matrix manipulation
*/

#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <random>

#include "TypedefinitionsAndGlobals.hh"

inline float addNormal(const float & value);

//Function to generate matrix with random entries distributed according to the normal distribution
Matrix RandomMatrix(const unsigned int & rowNum, const unsigned int & coloumnNum);
Vector RandomVector(const unsigned int & dimension);

//Functions to generate to vector corresponding to the correct output
//for example 3 -> (0,0,0,1,0,0,0,0,0,0)
Vector correctOutputVector(const char & correctOutputLabel); 

#endif