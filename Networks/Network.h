//
//  Network.h
//  deep learning
//
//  Created by Sébastien Raffray on 13/02/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Network__
#define __deep_learning__Network__

#include <vector>
#include "../Layers.h"
#include "../Utilities/Logger.h"
#include "../Utilities/Extractor.h"

class Network
{

public:
	Matrix *output_units;
	Matrix input_units;
	Logger logs;

	Network(cl_uint inputs);
	~Network();

	cl_uint get_type_of_layer(size_t layer_nbr);
	size_t get_nbr_layers();
	cl_uint get_nbr_inputs_in_layer(size_t layer_nbr);
	cl_uint get_nbr_outputs_in_layer(size_t layer_nbr);
	Matrix &get_weights_in_layer(size_t layer_nbr);
	Matrix &get_derivatives_in_layer(size_t layer_nbr);
	Matrix get_outputs_in_layer(size_t layer_nbr, const Matrix &input_data);
	void set_display(int mode = MODE_MIXED);

	void add_layer(int layer_type, cl_uint nbr_outputs, cl_float param = 0);
	void setup(cl_uint batch_size);
	void forward_pass(Matrix &input_data, Matrix &output_results);
	void backprop(const Matrix &inputs, const Matrix &labels);
	cl_float calculate_cost(const Matrix &label_data, const Matrix &result_data);
	cl_float calculate_error(const Matrix &label_data, const Matrix &result_data);
	Matrix reorder(Matrix &input_data, Matrix &label_data);
	void display(const Matrix &input_data);
	void save(string filename);
	void load(string filename);

private:
	int display_mode;
	cl_uint total_input_units;
	cl_uint total_output_units;
	//
public:
	//
	vector<Layer *> layers;

	void clear();
};

#endif /* defined(__deep_learning__Network__) */
