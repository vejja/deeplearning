//
//  ReLU.cpp
//  deep learning
//
//  Created by Fernando Raffray on 30/03/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "ReLU.h"

ReLU::ReLU() : Layer()
{
}

ReLU::ReLU(const cl_uint nbr_input_units, const cl_uint nbr_output_units) : Layer(nbr_input_units, nbr_output_units)
{
	layer_type = LAYER_RELU;
	layer_name = "ReLU";
}

ReLU::~ReLU()
{
}

void		ReLU::apply_activation(Matrix &outputs)
{
	outputs.apply_relu();
}

void		ReLU::apply_activation_derivatives(Matrix &outputs)
{
	outputs.apply_relu_deriv();
}
