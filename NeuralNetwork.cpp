/*
Created: 19/10/21
Author: d4bd
Object: Implementation of the neural network class
*/

#include <eigen3/Eigen/Eigen>       //Library for the algebric calculations
#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <random>
#include <fstream>

#include "Handler.hh"
#include "LoadSaveNetwork.hh"
#include "NeuralNetwork.hh"
#include "ActivationFunction.hh"
#include "DenseLayerOutput.hh"

//Class constructor
NeuralNetwork::NeuralNetwork(std::vector<unsigned int> const & topology, float learningRate, std::string fileName)
    : m_topology(topology), m_learningRate(learningRate), m_costFunctionValue(0), m_saveFile(fileName) //m_outputLayer(DenseLayerOutput( topology.at(topology.size()-2) , topology.at(topology.size()-1) )),
    {
        if(m_topology.size()<2)
        {
            std::cerr << "The network as only a layer" << std::endl;
            exit(4);
        }

        //Code for signal handler   
        struct sigaction sigIntHandler;

        sigIntHandler.sa_handler = sig_handler;
        sigemptyset(&sigIntHandler.sa_mask);
        sigIntHandler.sa_flags = 0;
        sigaction(SIGINT, &sigIntHandler, NULL);   

        //Let's build the actual neural network
        std::cout << "\nBuilding the neural network"; 

        //Checks if a file containing a neural network already exists since the weights and biases that charaterise the neural network can be saved (see saveNetwork funtion)
        if(std::filesystem::exists(m_saveFile))
        {
            loadSavedNetwork(m_saveFile, m_topology, m_hiddenLayers, m_outputLayer, m_learningEpoch, m_costFunctionValueHistory);
        }
        else
        {
            m_learningEpoch = 0;

            //Building the neural network from scratch
            //Adds hidden layers
            std::cout << " from scratch\n";
            for(std::vector<int>::size_type i = 1; i <= topology.size() - 2 ; ++i)
            {
                m_hiddenLayers.push_back(DenseLayer(topology.at(i-1),topology.at(i)));
            }
            //Adds output Layer
            m_outputLayer = DenseLayerOutput( topology.at(topology.size()-2) , topology.at(topology.size()-1) );
        }

        if(killed)
        {
            if(std::filesystem::exists(fileName))
                exit(SIGINT);
            else
            {
                saveNetwork(m_learningEpoch, m_topology, m_costFunctionValueHistory, m_hiddenLayers, m_outputLayer, m_saveFile);
                exit(SIGINT);
            }
        }
    }

void NeuralNetwork::propagateForward(const Matrix & input)                //Inputs are batches of inputs
{
    if(input.rows() > m_topology.at(0))
    {
        std::cerr << "Number of inputs is different from the ones expected. number of inputs: " << input.rows() << " , expected by the neural network: " << m_topology.at(0) << std::endl;
        exit(3);
    }
    //std::cout << "Neural Network calculation from input: " << input << "\n";

    //Propagates the input forward in the neural network via the sigmoid activation function
    //Calculation of the first hidden layer
    m_hiddenLayers.at(0).propagateForward(input);

    //Calculation of all the other hidden layers
    for(std::vector<int>::size_type i = 1; i < m_hiddenLayers.size(); i++)
    {   
        m_hiddenLayers.at(i).propagateForward(m_hiddenLayers.at(i-1).layerActivation());
    }

    //Calculation of the output layer
    m_outputLayer.propagateForward(m_hiddenLayers.back().layerActivation());

    //Output of the final output of the Neural Network which is just the output of the output layer after the activation function
    //std::cout << "Final output: " << m_outputLayer.layerActivation() << "\n";
}

void NeuralNetwork::costFunctionEvaluation(const Matrix & correctOutput, const bool & training)
{   
    Vector batchCostFunction;
    batchCostFunction.resize(m_outputLayer.layerActivation().cols());
    Matrix difference = m_outputLayer.layerActivation() - correctOutput;
    batchCostFunction = difference.colwise().squaredNorm()/2;
    m_costFunctionValue = batchCostFunction.sum();
    if(training)
        m_costFunctionValueHistory.push_back(m_costFunctionValue);
}

void NeuralNetwork::propagateBackward(const Matrix& correctOutput)
{
    m_outputLayer.errorCalculation(correctOutput);
    if(m_hiddenLayers.size() > 0)
    {
        m_hiddenLayers.back().setError(m_outputLayer.propagateBackward(m_hiddenLayers.back().layerWeightedInput()));
        for(std::vector<int>::size_type i = m_hiddenLayers.size()-1; i > 0 ; i--)
        {
            m_hiddenLayers.at(i-1).setError(m_hiddenLayers.at(i).propagateBackward(m_hiddenLayers.at(i-1).layerWeightedInput()));
        }
    }
}

void NeuralNetwork::SGD(const Matrix& input, const Matrix& correctOutput)
{
    propagateBackward(correctOutput);

    //Corrects wheight and biases for the output layer
    m_outputLayer.SGD(m_learningRate, m_hiddenLayers.back().layerActivation());
    if(m_hiddenLayers.size() > 0)
    {
        //Corrects wheight and biases for the first layer
        m_hiddenLayers.at(0).SGD(m_learningRate, input);
        //Corrects wheight and biases for all the other layers
        for(std::vector<int>::size_type i = 1; i < m_hiddenLayers.size(); i++)
        {
            m_hiddenLayers.at(i).SGD(m_learningRate, m_hiddenLayers.at(i-1).layerActivation());
        }
    }
}

void NeuralNetwork::trainCycle(std::vector<Vector> & inputs, std::vector<Vector> & correctOutputs, const unsigned int & batchSize)
{
    //Code for signal handler   
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
        
    //Calculates if the batchSize exactly divides the number of possible inputs
    unsigned int leftovers = inputs.size()%batchSize;

    //Input Matrix
    Matrix inputMatrix;
    inputMatrix.resize(inputs.at(0).rows(),batchSize);

    //Output Matrix
    Matrix outputMatrix;
    outputMatrix.resize(correctOutputs.at(0).rows(),batchSize);

    //trains the network with batches of the specified size, the leftovers form the exact division will be trated later
    for(std::vector<int>::size_type i = 0; i < inputs.size()-leftovers ; i += batchSize)
    {
        //Construction of the input matrix, i.e. the batch of inputs
        for(unsigned int j = 0; j < batchSize; j++)
        {
            inputMatrix.col(j) = inputs.at(i+j);
        }

        //Construction of the correct output matrix, i.e. the batch of correct outputs
        for(unsigned int j = 0; j < batchSize; j++)
        {
            outputMatrix.col(j) = correctOutputs.at(i+j);
        }

        //After the creation of the input and output matrix the input matrix is passed to the neural network for the feedForward algorithm
        propagateForward(inputMatrix);

        //Calculation of the errors for statistical purposes
        costFunctionEvaluation(outputMatrix, true);
        
        //Then we proceed to make the PGD
        SGD(inputMatrix, outputMatrix);
    }

    if(leftovers > 0)
    {
        //Construction of the input matrix, i.e. the batch of inputs
        inputMatrix.resize(inputs.at(inputs.size()-leftovers).rows(),leftovers);
        for(unsigned int j = 0; j < leftovers; j++)
        {
            inputMatrix.col(j) = inputs.at(inputs.size()-leftovers+j);
        }

        //Construction of the correct output matrix, i.e. the batch of correct outputs
        outputMatrix.resize(correctOutputs.at(correctOutputs.size()-leftovers).rows(),leftovers);
        for(unsigned int j = 0; j < leftovers; j++)
        {
            outputMatrix.col(j) = correctOutputs.at(correctOutputs.size()-leftovers+j);
        }

        //After the creation of the input and output matrix the input matrix is passed to the neural network for the feedForward algorithm
        propagateForward(inputMatrix);

        //Calculation of the errors for statistical purposes
        costFunctionEvaluation(outputMatrix, true);
        
        //Then we proceed to make the PGD
        SGD(inputMatrix, outputMatrix);  
    }
}

bool NeuralNetwork::outputCalculation(const Vector & correctOutput)
{
    Vector::Index maxRowCorrect, maxRowExpected, maxColExpected;
    [[maybe_unused]] float maxCorrect = correctOutput.maxCoeff(&maxRowCorrect);
    [[maybe_unused]] float maxExpected = m_outputLayer.layerActivation().maxCoeff(&maxRowExpected, &maxColExpected);

    //std::cout << "Expected output: " << maxRowCorrect << "\n";
    //std::cout << "Obtained output: " << maxRowExpected << "\n";
    costFunctionEvaluation(correctOutput, false);

    if(maxRowExpected == maxRowCorrect)
        return 1;
    else    
        return 0;

}

void NeuralNetwork::test(const std::vector<Vector> & checkInputs, const std::vector<Vector> & checkCorrectOutputs)
{
    //Code for signal handler   
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    //This function just needs to propagate forward the input and check if the output obtained through the neural network coincide with the expected one
    //Return the percentage of correct guesses
    unsigned int correctGuesses = 0;

    for(std::vector<int>::size_type i = 0; i < checkInputs.size() ; ++i)
    {      
        propagateForward(checkInputs.at(i));
        correctGuesses += outputCalculation(checkCorrectOutputs.at(i));
        //std::cout << correctGuesses << std::endl;
    }

    std::cout << "\n --- Percentage of correct guesses: " << float(correctGuesses)/(checkInputs.size())*100 << " ---\n\n";
}

void NeuralNetwork::train(std::vector<Vector> & inputs, std::vector<Vector> & correctOutputs, const unsigned int & batchSize, const unsigned int & epoch, const std::vector<Vector> & checkInputs, const std::vector<Vector> & checkCorrectOutputs)
{
    if(m_learningEpoch < epoch)
    {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

        //Create two random engines with the same state
        std::mt19937 eng1(seed);
        auto eng2 = eng1;

        std::cout << "\nTraining the Neural Network\n";

        for(unsigned int i = m_learningEpoch; i < epoch; i++)
        {
            std::cout << "Epoch: " << i << "\n";

            std::shuffle(begin(inputs), end(inputs), eng1);
            std::shuffle(begin(correctOutputs), end(correctOutputs), eng2);

            trainCycle(inputs, correctOutputs, batchSize);
            test(checkInputs, checkCorrectOutputs);

            m_learningEpoch++;
            if(killed)
            {
                saveNetwork(m_learningEpoch, m_topology, m_costFunctionValueHistory, m_hiddenLayers, m_outputLayer, m_saveFile);
                exit(SIGINT);
            }
        }
        std::cout << "End of training\n\n";

        saveNetwork(m_learningEpoch, m_topology, m_costFunctionValueHistory, m_hiddenLayers, m_outputLayer, m_saveFile);
    }
    else 
    {
        std::cout << "Neural Network already completed training\n";
        test(checkInputs, checkCorrectOutputs);
    }
}