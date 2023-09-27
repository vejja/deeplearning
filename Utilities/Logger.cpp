//
//  Logger.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 10/04/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Logger.h"

Logger::Logger()
{
	// Opens a new log file
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::stringstream filenamestream;
	filenamestream << "Logs/" << std::put_time(&tm, "%Y-%m-%d %H-%M-%S") << ".log";
	logfile.open(filenamestream.str(), std::ios::app);
}

Logger::~Logger()
{
	// Closes the log file
	// No need to close manually as logfile destructor does it already
}

void	Logger::add(std::string logstring)
{
	// Appends a log entry in the file
	std::cout << "[LOG] : " << logstring << std::endl;
	logfile << logstring << std::endl;
}