/*
Created: 30/11/21
Author: Space
Object: File to store all user type definitions and global variables
*/

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <eigen3/Eigen/Eigen>

//Series of typedef for easier readability of the program
using Matrix = Eigen::MatrixXf;                 //Matrix of float of arbitrary dimensions
using Vector = Eigen::VectorXf;                 //Coloumn vector of float of arbitray dimension

//Global boolean variable for signal handling
extern bool killed;

#endif