//
//  Softmax.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 13/02/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Softmax.h"

Softmax::Softmax() : Layer()
{
}

Softmax::Softmax(const cl_uint nbr_visible_units, const cl_uint nbr_hidden_units) : Layer(nbr_visible_units, nbr_hidden_units)
{
	layer_type = LAYER_SOFTMAX;
	layer_name = "Softmax";
}

Softmax::~Softmax()
{
}

void		Softmax::apply_activation(Matrix &outputs)
{
	outputs.apply_softmax();
}

cl_float	Softmax::calculate_cost(const Matrix &label_data, const Matrix &result_data) const
{
	// Cross-entropy function is used as cost function for Softmax Layer
	// -SUMj (Tj log yj)
	return Matrix::cross_entropy(label_data, result_data);
}

cl_float	Softmax::calculate_error(const Matrix &label_data, const Matrix &result_data) const
{
	return Matrix::error_rate(label_data, result_data);
}

void		Softmax::deriv_sumprod(Layer *previous_layer)
{
	// Override la fonction de base pour la derniere couche
	// output_units doit etre pre-loade avec output_units - labels

	// Ignore la couche precedente (il n'y en a pas) et initialise les deltas 
	deltas.is_deep_copy_of(output_units);
	// Neutralise pour la suite le calcul des derivées sur la première colonne des neurones
	deltas.fill_left_column(0);
}

