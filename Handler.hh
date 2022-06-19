/*
Created: 30/10/21
Author: Space
Object: Definition of the signal handling function
*/

#ifndef HANDLER_H 
#define HANDLER_H

#include <iostream>
#include <csignal>

#include "TypedefinitionsAndGlobals.hh"

void sig_handler([[maybe_unused]] int s);

#endif