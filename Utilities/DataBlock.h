//
//  DataBlock.h
//  deep learning
//
//  Created by SŽbastien Raffray on 27/01/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__DataBlock__
#define __deep_learning__DataBlock__

#include <iostream>
#include <fstream>
#include <random>
// includes below used for the ntohl function
#ifdef _WIN32
#include <winsock2.h>
#elif linux
#include <netinet/in.h>
#endif

using namespace std;

class DataBlock
{
public:
    float			*cpu_buffer;
	unsigned int	nb_rows;
	unsigned int	nb_cols;

	DataBlock();
	DataBlock(unsigned int nbr_rows, unsigned int nbr_columns);
	~DataBlock();
    
    DataBlock(const DataBlock& image);
    DataBlock(DataBlock&& image);
    DataBlock& operator=(DataBlock image);

    unsigned int size() const;
	void		strip_left_column();
    void		insert_left_column(float value);
	DataBlock	extract_row(unsigned int row_nbr) const;
	DataBlock	extract_column(unsigned int col_nbr) const;
	void		add_scalar(float scalar);
	void		multiply_by_scalar(float scalar);
	void		fill_with_random_gaussian(const float mean, const float stdev);
	void		fill_with_random_uniform(const float lo, const float hi);
	void		fill_with_random_binary(const float prob);
    
    friend void swap(DataBlock& first, DataBlock& second);

private:
	static default_random_engine generator;
};

#endif /* defined(__deep_learning__DataBlock__) */