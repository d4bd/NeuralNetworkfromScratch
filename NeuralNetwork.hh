/*
Created: 19/10/21
Author: Space
Object: Header file for the neural network class
*/

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <eigen3/Eigen/Eigen>       //Library for the algebric calculations
#include <vector>
#include <string>
#include <fstream>

#include "DenseLayer.hh"
#include "DenseLayerOutput.hh"

//Rendere la classe singleton

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<unsigned int> const & topology, float learningRate, std::string fileName = "neuralNetwork.txt");

    //Selector methods
    std::vector<unsigned int> neuralNetworkTopopogy() const {return m_topology;};
    float neuralNetworkLearningRate() const {return m_learningRate;};

    //Modifier methods
    void propagateForward(const Matrix & input);                                        //Returns the output of the neurons' dense layer  
    void costFunctionEvaluation(const Matrix & correctOutput, const bool & training);   //Calculates the output error  
    void propagateBackward(const Matrix & correctOutput);
    void SGD(const Matrix & input, const Matrix & correctOutput);                       //Updates weights and biases according to PGD, as prescribed by the mini batch approach taken to train the network    

    //Function to train the network as inputs accepts a vector of all the possible inputs of trainin with the vector containing the expected result for each input
    //The input batches are created during the training process and the size can be specified with the parameter batchSize
    void trainCycle(std::vector<Vector> & inputs, std::vector<Vector> & correctOutputs, const unsigned int & batchSize); 

    //Function that only checks if the obtained output coincides with the expected one, if they coincide it returns 1 otherwise it returns zero
    bool outputCalculation(const Vector & correctOutput);    
    //Function to test the neural network as input accepts a vector of all the possible inputs for the testing with the vector containing the expected result for each input
    void test(const std::vector<Vector> & checkInputs, const std::vector<Vector> & checkCorrectOutputs);

    void train(std::vector<Vector> & inputs, std::vector<Vector> & correctOutputs, const unsigned int & batchSize, const unsigned int & epoch, const std::vector<Vector> & checkInputs, const std::vector<Vector> & checkCorrectOutputs);

private:
    std::vector<unsigned int> m_topology;           //Vector will store configuration of the neural network, i.e topology = {1,3,5,2} will indicate a neural network of 4 layers, the input one with 1 neuron, the output one with 2 neaurons and the two hidden ones with 3 and 5 neurons 
    float m_learningRate;                           //Increment for the PGD
    std::vector<DenseLayer> m_hiddenLayers;         //Vector containing the hidden layers of the neural network
    DenseLayerOutput m_outputLayer;                 //Output layer of the neural network
                                                    //There is no input layer, since is basically a layer with the identity as weight matrix and zero bias, the input will just be passed to the function propagateForward that evolves the neural network
    float m_costFunctionValue;                      //Mean of all the cost function value obtained from a batch
    std::vector<float> m_costFunctionValueHistory;  //Vector containing the mean loss obtained through each iteration of the learning process to have a global view of the progress of the network    
    
    unsigned int m_learningEpoch;                   //Starting cycle for the learning  

    std::string m_saveFile;
};


#endif