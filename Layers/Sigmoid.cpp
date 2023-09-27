//
//  Sigmoid.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 09/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Sigmoid.h"

Sigmoid::Sigmoid() : Layer()
{
}

Sigmoid::Sigmoid(const cl_uint nbr_input_units, const cl_uint nbr_output_units) : Layer(nbr_input_units, nbr_output_units)
{
	layer_type = LAYER_SIGMOID;
	layer_name = "Sigmoid";
}

Sigmoid::~Sigmoid()
{
}

void Sigmoid::apply_activation(Matrix &outputs)
{
	outputs.apply_sigmoid();
}

void Sigmoid::apply_activation_derivatives(Matrix &outputs)
{
	outputs.apply_sigmoid_deriv();
}
