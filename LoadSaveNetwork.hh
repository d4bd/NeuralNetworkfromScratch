/*
Created: 30/10/21
Author: Space
Object: Declaration of functions to save and load a neural network
*/

#ifndef WRITEREAD_H
#define WRITEREAD_H

#include <vector>
#include <string>

#include "DenseLayerOutput.hh"

void saveNetwork(const int & learningEpoch,
                        const std::vector<unsigned int> & topology,
                        const std::vector<float> & costFunctionValueHistory,
                        const std::vector<DenseLayer> & hiddenLayers,
                        const DenseLayerOutput & outputLayer,
                        const std::string & fileName);

void loadSavedNetwork(const std::string & fileName, 
                    const std::vector<unsigned int> & topology, 
                    std::vector<DenseLayer> & hiddenLayers, 
                    DenseLayerOutput & outputLayer, 
                    unsigned int & learningEpoch, 
                    std::vector<float> & costFunctionValueHistory);

#endif