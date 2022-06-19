/*
Created: 30/10/21
Author: Space
Object: Implementation of the function to read and load the MNIST data
*/

#include <fstream>
#include <iostream>

#include "ReadMinst.hh"
#include "Handler.hh"

void read_mnist_cv(std::vector<Vector> & inputs, 
                std::vector<Vector> & correctOutputs, 
                const std::string & image_filename, 
                const std::string & label_filename)
{
    //Code for signal handler   
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = sig_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2051){
        std::cout<<"Incorrect image file magic: "<<magic<<"\n";
        return;
    }

    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if(magic != 2049){
        std::cout<<"Incorrect label file magic: "<<magic<<"\n";
        return;
    }

    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    if(num_items != num_labels){
        std::cout<<"image file nums should equal to label num"<<"\n";
        return;
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    //std::cout<<"Image and label num is: "<<num_items<<"\n";
    //std::cout<<"Image rows: "<<rows<<", cols: "<<cols<<"\n";

    char label;
    char pixel;

    Vector input;
    input.resize(rows * cols);

    for (uint32_t item_id = 0; item_id < num_items; ++item_id) 
    {
        //Read image pixel
        for (uint32_t i = 0; i < rows * cols; i++)
        {
            image_file.read(&pixel, 1);
            input(i) = float(inverse(int(pixel)))/255;
        }
        inputs.push_back(input);

        //Read label
        label_file.read(&label, 1);
        correctOutputs.push_back(correctOutputVector(label));
          
        if(killed)
        {
            image_file.close();
            label_file.close();
            exit(SIGINT);
        }
    }

    image_file.close();
    label_file.close();

    std::cout << "\nAll input data collected from " << image_filename << " and " << label_filename << "\n";
}