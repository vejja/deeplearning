//
//  Adam.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 15/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "Adam.h"

Adam::Adam(cl_float init_learning_rate,
		   cl_float init_gradient_decay,
		   cl_float init_squares_decay,
		   cl_float damping_factor,
		   cl_float decay_scaling) : Optimizer()
{
	optimizer_name = "Adam";
	nbr_iter = 0;
	alpha = init_learning_rate;
	beta1 = init_gradient_decay;
	beta2 = init_squares_decay;
	epsilon = damping_factor;
	lambda = decay_scaling;
}

Adam::~Adam()
{
}

Adam&	Adam::with_init_learning_rate(cl_float init_learning_rate)
{
	alpha = init_learning_rate;
	return *this;
}

Adam&	Adam::with_init_gradient_decay(cl_float init_gradient_decay)
{
	beta1 = init_gradient_decay;
	return *this;
}

Adam&	Adam::with_init_squares_decay(cl_float init_squares_decay)
{
	beta2 = init_squares_decay;
	return *this;
}

Adam&	Adam::with_damping_factor(cl_float damping_factor)
{
	epsilon = damping_factor;
	return *this;
}

Adam&	Adam::with_decay_scaling(cl_float decay_scaling)
{
	lambda = decay_scaling;
	return *this;
}

void	Adam::initialise(cl_uint mini_batch_size,
							Network &network)
{
	Optimizer::initialise(mini_batch_size, network);
	for (size_t i = 0; i < biased_avg_gradients.size(); i++) {
		delete biased_avg_gradients[i];
	}
	for (size_t i = 0; i < biased_avg_squares.size(); i++) {
		delete biased_avg_squares[i];
	}
	biased_avg_gradients.clear();
	biased_avg_squares.clear();
	biased_avg_gradients.resize(get_nbr_layers());
	biased_avg_squares.resize(get_nbr_layers());
	for (size_t i = 0; i < get_nbr_layers(); i++) {
		biased_avg_gradients[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
			get_nbr_outputs_in_layer(i) + 1);
		*biased_avg_gradients[i] = 0.0f;
		biased_avg_squares[i] = new Matrix(get_nbr_inputs_in_layer(i) + 1,
			get_nbr_outputs_in_layer(i) + 1);
		*biased_avg_squares[i] = 0.0f;
	}
}

void	Adam::optimize(Matrix &inputs,
						  Matrix &outputs)
{
	cl_float alpha_t;
	// t <- t+1
	nbr_iter++;
	// beta(1,t) <- beta1 * lambda ^ (t-1)
	if (nbr_iter == 1)
		beta1_t = beta1;
	else
		beta1_t *= lambda;
	// g(t) = Gradient ft(theta(t-1))
	m_network->backprop(inputs, outputs);
	// alpha(t) = alpha . sqrt(1 - beta2^t) / (1 - beta1^t)
	alpha_t = alpha * sqrt(1 - pow(beta2, nbr_iter)) / (1 - pow(beta1, nbr_iter));
	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		Matrix &weights = get_weights_in_layer(i);
		Matrix &derivatives = get_derivatives_in_layer(i);
		Matrix &avg_gradients = *biased_avg_gradients[i];
		Matrix &avg_squares = *biased_avg_squares[i];
		if (derivatives.nb_elements()) {
			// m(t) <- beta(1,t).m(t-1) + (1 -beta(1,t)).g(t)
			avg_gradients *= beta1_t;
			avg_gradients.add_scaled(derivatives, 1 - beta1_t);
			// v(t) <- beta2.v(t-1) + (1 - beta2).g(t)^2
			derivatives *= derivatives;
			avg_squares *= beta2;
			avg_squares.add_scaled(derivatives, 1 - beta2);
			// theta(t) <- theta(t-1) - alpha(t) . m(t)/(sqrt(v(t)) + epsilon)
			weights.add_with_rms_scaling(avg_gradients, -alpha_t, avg_squares, epsilon);
		}
	}
}