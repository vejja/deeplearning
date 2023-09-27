//
//  Logger.h
//  deep learning
//
//  Created by Sebastien Raffray on 10/04/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Logger__
#define __deep_learning__Logger__

#include <fstream>	// ofstream
#include <sstream>  // stringstream
#include <string>	// << a string into an ofstream
#include <ctime>	// std::time
#include <iomanip>	// std::put_time
#include <iostream> // std::cout


class Logger {

public:
	Logger();
	~Logger();

	void add(std::string logstring);

private:
	std::ofstream logfile;
};

#endif /* defined(__deep_learning__Logger__) */
