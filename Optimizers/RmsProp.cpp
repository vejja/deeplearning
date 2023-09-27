//
//  RmsProp.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 04/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "RmsProp.h"

RmsProp::RmsProp(cl_float learning_rate,
				 cl_float decay_rate,
				 cl_float damping_factor) : Optimizer()
{
	optimizer_name = "RmsProp";
	m_learning_rate = learning_rate;
	m_decay_rate = decay_rate;
	m_damping_factor = damping_factor;
}

RmsProp::~RmsProp()
{
}


void	RmsProp::initialise(cl_uint mini_batch_size,
							Network &network)
{
	Optimizer::initialise(mini_batch_size, network);
	for (size_t i = 0; i < m_mean_squares.size(); i++) {
		delete m_mean_squares[i];
	}
	m_mean_squares.clear();
	m_mean_squares.resize(get_nbr_layers());
	for (size_t i = 0; i < get_nbr_layers(); i++) {
		m_mean_squares[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
									   get_nbr_outputs_in_layer(i) + 1);
		*m_mean_squares[i] = 0.0f;
	}
}

void	RmsProp::optimize(Matrix &inputs,
						  Matrix &outputs)
{
	m_network->backprop(inputs, outputs);
	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		Matrix &weights = get_weights_in_layer(i);
		Matrix &derivatives = get_derivatives_in_layer(i);
		Matrix &mean_squares = *m_mean_squares[i];
		if (derivatives.nb_elements()) {
			mean_squares.apply_mean_square_update(m_decay_rate, derivatives);
			weights.add_with_rms_scaling(derivatives, -m_learning_rate, mean_squares, m_damping_factor);
		}
	}
}