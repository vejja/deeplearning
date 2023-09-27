//
//  RmsNesterov.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 15/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "RmsNesterov.h"

RmsNesterov::RmsNesterov(cl_float start_learning_rate,
						 cl_float decay_rate,
						 cl_float damping_factor,
						 cl_float momentum,
						 cl_float min_learning_rate,
						 cl_float max_learning_rate,
						 cl_float rate_adapt) : Optimizer()
{
	optimizer_name = "RmsNesterov";
	m_start_learning_rate = start_learning_rate;
	m_decay_rate = decay_rate;
	m_damping_factor = damping_factor;
	m_momentum = momentum;
	m_min_learning_rate = min_learning_rate;
	m_max_learning_rate = max_learning_rate;
	m_rate_adapt = rate_adapt;
}

RmsNesterov::~RmsNesterov()
{
}

RmsNesterov&	RmsNesterov::with_start_learning_rate(cl_float start_learning_rate)
{
	m_start_learning_rate = start_learning_rate;
	return *this;
}

RmsNesterov&	RmsNesterov::with_decay_rate(cl_float decay_rate)
{
	m_decay_rate = decay_rate;
	return *this;
}

RmsNesterov&	RmsNesterov::with_damping_factor(cl_float damping_factor)
{
	m_damping_factor = damping_factor;
	return *this;
}

RmsNesterov&	RmsNesterov::with_momentum(cl_float momentum)
{
	m_momentum = momentum;
	return *this;
}

RmsNesterov&	RmsNesterov::with_min_learning_rate(cl_float min_learning_rate)
{
	m_min_learning_rate = min_learning_rate;
	return *this;
}

RmsNesterov&	RmsNesterov::with_max_learning_rate(cl_float max_learning_rate)
{
	m_max_learning_rate = max_learning_rate;
	return *this;
}

RmsNesterov&	RmsNesterov::with_rate_adapt(cl_float rate_adapt)
{
	m_rate_adapt = rate_adapt;
	return *this;
}

void	RmsNesterov::initialise(cl_uint mini_batch_size,
							Network &network)
{
	Optimizer::initialise(mini_batch_size, network);
	for (size_t i = 0; i < m_mean_squares.size(); i++)
	{
		delete m_mean_squares[i];
	}
	for (size_t i = 0; i < m_velocity.size(); i++)
	{
		delete m_velocity[i];
	}
	for (size_t i = 0; i < m_learning_rates.size(); i++)
	{
		delete m_learning_rates[i];
	}
	m_mean_squares.clear();
	m_velocity.clear();
	m_learning_rates.clear();
	
	m_mean_squares.resize(get_nbr_layers());
	m_velocity.resize(get_nbr_layers());
	m_learning_rates.resize(get_nbr_layers());
	
	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		m_mean_squares[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
									   get_nbr_outputs_in_layer(i) + 1);
		*m_mean_squares[i] = 0.0f;
		m_velocity[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
								   get_nbr_outputs_in_layer(i) + 1);
		*m_velocity[i] = 0.0f;
		m_learning_rates[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
									  get_nbr_outputs_in_layer(i) + 1);
		*m_learning_rates[i] = m_start_learning_rate;
	}
}

void	RmsNesterov::optimize(Matrix &inputs,
							  Matrix &outputs)
{
	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		Matrix &weights = get_weights_in_layer(i);
		Matrix &velocity = *m_velocity[i];
		if (get_derivatives_in_layer(i).nb_elements()) {
			velocity *= m_momentum;
			weights -= velocity;
		}
	}
	m_network->backprop(inputs, outputs);
	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		Matrix &weights = get_weights_in_layer(i);
		Matrix &derivatives = get_derivatives_in_layer(i);
		Matrix &mean_squares = *m_mean_squares[i];
		Matrix &velocity = *m_velocity[i];
		Matrix &learning_rates = *m_learning_rates[i];
		if (derivatives.nb_elements()) {
			mean_squares.apply_mean_square_update(m_decay_rate, derivatives);
			derivatives.apply_rms_scaling(learning_rates, mean_squares, m_damping_factor);
			learning_rates.adapt_and_clip(derivatives, velocity, m_rate_adapt, m_min_learning_rate, m_max_learning_rate);
			velocity += derivatives;
			weights -= derivatives;
//			velocity.display("velocity");
//			derivatives.display("derivatives");
//			learning_rates.display("learning rates");
		}
	}
}