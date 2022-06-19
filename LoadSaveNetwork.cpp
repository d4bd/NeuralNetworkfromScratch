/*
Created: 30/10/21
Author: Space
Object: Implemetation of functions to save and load a neural network
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#include "LoadSaveNetwork.hh"

void saveNetwork(const int & learningEpoch,
                        const std::vector<unsigned int> & topology,
                        const std::vector<float> & costFunctionValueHistory,
                        const std::vector<DenseLayer> & hiddenLayers,
                        const DenseLayerOutput & outputLayer,
                        const std::string & fileName)
{
    std::cout << "Saving neural network obtained after " << learningEpoch << " epochs\n";
    std::ofstream outFile (fileName);

    if(outFile.is_open())
    {
        //Saves the topology of the neural network 
        outFile << "Topology\n";
        std::ostream_iterator<int> output_iterator(outFile, " ");
        std::copy(topology.begin(), topology.end(), output_iterator);
        outFile << "\n";

        //Saves the element to which the learning algorithm was arrived form the pool of all the inputs
        outFile << "Learing epoch\n" << learningEpoch << "\n";

        //Saves the history of mean losses of each learning cycle for statistical purposes
        outFile << "Cost function values\n";
        std::ostream_iterator<float> output_iterator1(outFile, " ");
        std::copy(costFunctionValueHistory.begin(), costFunctionValueHistory.end(), output_iterator1);
        outFile << "\n";

        //Saves all the layers' weights and biases
        for(std::vector<int>::size_type i = 1; i <= hiddenLayers.size(); i++)
        {   
            outFile << "Layer: " << i << "\n";
            outFile << "Weights\n";
            outFile << hiddenLayers.at(i-1).layerWeights() << "\n"; 
            outFile << "Biases\n";
            outFile << hiddenLayers.at(i-1).layerBiases() << "\n"; 
        }
        outFile << "Output Layer\n";
        outFile << "Weights\n";
        outFile << outputLayer.layerWeights() << "\n"; 
        outFile << "Biases\n";
        outFile << outputLayer.layerBiases() << "\n"; 
    } 

    outFile.close();
}

void loadSavedNetwork(const std::string & fileName, 
                    const std::vector<unsigned int> & topology, 
                    std::vector<DenseLayer> & hiddenLayers, 
                    DenseLayerOutput & outputLayer, 
                    unsigned int & learningEpoch, 
                    std::vector<float> & costFunctionValueHistory)
{
    std::cout << " from file\n";
    //Building the nueral network from data of a preceding training of the network
    std::ifstream inFile(fileName);
    if(inFile.is_open())
    {
        Matrix weightsTemp;
        Vector biasesTemp;
        unsigned int layerNum = 0;  //To keep track of the layer we are working on
        for( std::string line; getline(inFile, line); )
        {
            if(line == "Topology")
            {
                getline(inFile, line);
                //Converts string of numbers into vector
                std::stringstream iss(line);
                unsigned int layerNeurons;
                std::vector<unsigned int> oldTopology;
                while(iss >> layerNeurons)
                    oldTopology.push_back(layerNeurons);

                if(topology != oldTopology)
                {
                    std::cerr << "The topology of the neural network is inconpatible with the topology of the neural network that you are trying to load through the file\n";
                    exit(1);
                }
            }
            else if(line == "Learing epoch")
            {
                getline(inFile, line);
                learningEpoch = std::stoi(line);
            }
            else if(line == "Cost function values")
            {
                getline(inFile, line);
                //Converts string of numbers into vector
                std::stringstream iss(line);
                float costFunctionValue;
                while(iss >> costFunctionValue)
                    costFunctionValueHistory.push_back(costFunctionValue);
            }
            else if(line.find("Layer") != std::string::npos)
            {
                layerNum++; 
                int rowNum = topology.at(layerNum);
                int colNum = topology.at(layerNum-1);
                getline(inFile, line); 
                if(line == "Weights")
                {
                    //Creates the weights matrix
                    //First the matrix is resized to the correct dimensions
                    weightsTemp.resize(rowNum,colNum); 

                    //We get each row of the matrix one by one (row of matrix == line of file), after the conversion from string to vector of float each element is inserted in the matrix via the element position (i,j) in the matrix 
                    for(int i = 0; i < rowNum; i++)
                    {
                        getline(inFile, line); 

                        //Converts string of numbers into vector
                        std::stringstream iss( line );
                        float number;
                        std::vector<float> myNumbers;
                        while ( iss >> number )
                            myNumbers.push_back( number );

                        //Adds each element to the matrix
                        for(int j = 0; j < colNum; j++)
                            weightsTemp(i,j) = myNumbers.at(j);
                    } 
                }
                getline(inFile, line); 
                if(line == "Biases")
                {
                    //Creates the biases vector
                    //First the vector is resized to the correct dimension
                    biasesTemp.resize(rowNum);

                    //We get each element of the vector as a line of the file, after the conversion from string to float each element is inserted in the vector
                    for(int i = 0; i < rowNum; i++)
                    {
                        getline(inFile, line); 

                        float myNumber = std::stof(line);
                        biasesTemp(i) = myNumber;
                    }        
                }
                
                if(layerNum == topology.size()-1)
                {
                    outputLayer = DenseLayerOutput(topology.at(layerNum-1),topology.at(layerNum),weightsTemp,biasesTemp);
                }   
                else
                {
                    hiddenLayers.push_back(DenseLayer(topology.at(layerNum-1),topology.at(layerNum),weightsTemp,biasesTemp));
                }
            } 
        }
    }
    inFile.close();
}