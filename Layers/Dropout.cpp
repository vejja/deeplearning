//
//  Dropout.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 02/04/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Dropout.h"

Dropout::Dropout()
{
}

Dropout::Dropout(cl_uint nbr_input_and_output_units, cl_float init_retain_rate)
		: Layer()
{
	retain_rate = init_retain_rate;
	nbr_inputs = nbr_input_and_output_units;
	nbr_outputs = nbr_input_and_output_units;
	layer_type = LAYER_DROPOUT;
	layer_name = "Dropout";
}

Dropout::~Dropout()
{
}

cl_float Dropout::get_retain_rate() const
{
	return retain_rate;
}

void Dropout::forward_pass(Matrix &inputs, Matrix &outputs)
{
	// Dropout in generative mode : averages many subnets in one pass
	outputs.is_deep_copy_of(inputs);
	outputs *= retain_rate;
}

void Dropout::setup(cl_uint mini_batch_size,
										Matrix *lower_outputs)
{
	// sets-up mask
	mask = Matrix(mini_batch_size, nbr_outputs + 1);
	// no derivatives
	// no input_units
	// output units are copied from the lower layer
	output_units.is_shallow_copy_of(*lower_outputs);
}

void Dropout::backprop_up()
{
	// Dropout in training mode : samples one subnet per training case in the mini-batch pass
	mask.fill_with_random_binary(retain_rate);
	output_units *= mask;
}

void Dropout::backprop_down(Layer *upper_layer)
{
	// deltas are handed over from upper layer
	deltas.is_shallow_copy_of(upper_layer->deltas);
	// weights are handed over from upper layer also
	weights.is_shallow_copy_of(upper_layer->weights);
	// Should hand over mask also, but not required for sigmoid and relu
}
