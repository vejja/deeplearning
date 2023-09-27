//
//  DataBlock.h
//  deep learning
//
//  Created by SŽbastien Raffray on 29/01/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "DataBlock.h"

default_random_engine DataBlock::generator/*((unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count())*/;

DataBlock::DataBlock() 
{
	cpu_buffer = nullptr;
	nb_rows = 0;
	nb_cols = 0;
}

DataBlock::DataBlock(unsigned int nbr_rows, unsigned int nbr_columns) : DataBlock()
{
	cpu_buffer = new float[nbr_rows * nbr_columns];
	nb_rows = nbr_rows;
	nb_cols = nbr_columns;
}


DataBlock::~DataBlock()
{
    delete[] cpu_buffer;
}

DataBlock::DataBlock(const DataBlock& image) : DataBlock(image.nb_rows, image.nb_cols)
{
    for (unsigned int i = 0; i < image.size(); i++) {
        cpu_buffer[i] = image.cpu_buffer[i];
    }
}

DataBlock::DataBlock(DataBlock&& image) : DataBlock()
{
    swap(*this, image);
}

DataBlock&	DataBlock::operator=(DataBlock image)
{
    swap(*this, image);
    return(*this);
}


unsigned int DataBlock::size() const
{
    return nb_rows * nb_cols;
}

void	DataBlock::strip_left_column()
{
	DataBlock temp(nb_rows, nb_cols - 1);
	size_t temp_pos = 0;
	size_t src_pos = 0;
	for (unsigned int cur_row = 0; cur_row < nb_rows; cur_row++) {
		src_pos++;
		for (unsigned int cur_col = 1; cur_col < nb_cols; cur_col++) {
			temp.cpu_buffer[temp_pos] = cpu_buffer[src_pos];
			temp_pos++;
			src_pos++;
		}
	}
	swap(*this, temp);
}

void    DataBlock::insert_left_column(float value)
{
    DataBlock temp(nb_rows, nb_cols + 1);
    size_t temp_pos = 0;
    size_t src_pos = 0;
    for (unsigned int cur_row = 0; cur_row < nb_rows; cur_row++) {
        temp.cpu_buffer[temp_pos] = value;
        temp_pos++;
        for (unsigned int cur_col = 0; cur_col < nb_cols; cur_col++) {
            temp.cpu_buffer[temp_pos] = cpu_buffer[src_pos];
            temp_pos++;
            src_pos++;
        }
    }
    swap(*this, temp);
}

DataBlock	DataBlock::extract_row(unsigned int row_nbr) const
{
	DataBlock line(1, nb_cols);
	for (unsigned int cur_col = 0; cur_col < nb_cols; cur_col++) {
		line.cpu_buffer[cur_col] = cpu_buffer[(row_nbr * nb_cols) + cur_col];
	}
	return line;
}

DataBlock	DataBlock::extract_column(unsigned int col_nbr) const
{
	DataBlock line(1, nb_rows);
	for (unsigned int cur_row = 0; cur_row < nb_rows; cur_row++) {
		line.cpu_buffer[cur_row] = cpu_buffer[(cur_row * nb_cols) + col_nbr];
	}
	return line;
}

void	DataBlock::add_scalar(float scalar)
{
	for (unsigned int i = 0; i < size(); i++) {
		cpu_buffer[i] += scalar;
	}
}

void	DataBlock::multiply_by_scalar(float scalar)
{
	for (unsigned int i = 0; i < size(); i++) {
		cpu_buffer[i] *= scalar;
	}
}

void	DataBlock::fill_with_random_gaussian(const float mean, const float stdev)
{
	normal_distribution<float> distribution(mean, stdev);
	for (unsigned int i = 0; i < size(); i++) {
		cpu_buffer[i] = distribution(generator);
	}
}

void	DataBlock::fill_with_random_uniform(const float lo, const float hi)
{
	uniform_real_distribution<float> distribution(lo, hi);
    for (unsigned int i = 0; i < size(); i++) {
        cpu_buffer[i] = distribution(generator);
    }
}

void	DataBlock::fill_with_random_binary(const float prob)
{
	uniform_real_distribution<float> distribution(0, 1);
	
	for (unsigned int i = 0; i < size(); i++) {
		if (distribution(generator) <= prob)
			cpu_buffer[i] = 1.0f;
		else
			cpu_buffer[i] = 0.0f;
	}
}

void    swap(DataBlock& first, DataBlock& second)
{
    swap(first.cpu_buffer, second.cpu_buffer);
    swap(first.nb_rows, second.nb_rows);
    swap(first.nb_cols, second.nb_cols);
}


