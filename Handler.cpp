/*
Created: 30/10/21
Author: Space
Object: Implementation of the signal handling function
*/

#include "Handler.hh"

void sig_handler([[maybe_unused]] int s)
{
    std::cerr << "\n\n --- Called termination of the program, terminating calculation, saving the network and closing files ---\n\n";
    killed = true;
}