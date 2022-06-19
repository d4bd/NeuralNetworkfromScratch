/*
Created: 30/10/21
Author: Space
Object: Definition of the function to read and load the MNIST data
*/

#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <vector>
#include <string>

#include "Matrix.hh"

inline int inverse(int num)
{
    return (num >= 0) ? num : num+256;
}

inline uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_mnist_cv(std::vector<Vector> & inputs, 
                std::vector<Vector> & correctOutputs, 
                const std::string & image_filename, 
                const std::string & label_filename);

#endif