//
//  Layer.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 03/03/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "Layer.h"

Layer::Layer()
{
	displayer = nullptr;
	input_units = nullptr;
	layer_type = LAYER_UNDEFINED;
}

Layer::Layer(cl_uint nbr_visible_units,
						 cl_uint nbr_hidden_units,
						 cl_float mean,
						 cl_float stdev) : Layer()
{
	nbr_inputs = nbr_visible_units;
	nbr_outputs = nbr_hidden_units;
	weights = Matrix(nbr_inputs + 1, nbr_outputs + 1);
	/*
	// start by filling the table with gaussian random numbers
	weights.fill_with_random_gaussian(0.0f, 0.01f);
	// init visible biases to 0
	weights.fill_left_column(0);
	// init hidden biases to 0
	weights.fill_top_row(0);
	*/
	// initializes the weights with the Xavier procedure
	cl_float boundary = sqrtf(6.0f / (nbr_visible_units + nbr_hidden_units));
	weights.fill_with_random_uniform(-boundary, boundary);

	// init visible biases to 0
	// weights.fill_left_column(0);
	// init hidden biases to 0
	// weights.fill_top_row(0);
}

Layer::~Layer()
{
	delete displayer;
}

cl_uint Layer::get_type() const
{
	return layer_type;
}

string Layer::get_name() const
{
	return layer_name;
}

cl_uint Layer::get_nbr_inputs() const
{
	return nbr_inputs;
}

cl_uint Layer::get_nbr_outputs() const
{
	return nbr_outputs;
}

string Layer::get_title() const
{
	return displayer->get_title();
}

void Layer::init_weights(const Matrix &input_images)
{
	// init visible biases to log(p/(1-p)) instead of plain zeroes
	if (input_images.get_cols() != nbr_inputs + 1)
		throw std::runtime_error("Layer - wrong size of input_images Matrix for initialization of weights");
	weights.left_is_avg_pxls(input_images);
	weights.fill_top_row(0);
}

void Layer::forward_pass(Matrix &inputs, Matrix &outputs)
{
	inputs.fill_left_column(1);
	outputs.is_mult_of(inputs, weights);
	apply_activation(outputs);
}

void Layer::setup(cl_uint mini_batch_size, Matrix *lower_outputs)
{
	deltas = Matrix(mini_batch_size, nbr_outputs + 1);
	derivatives = Matrix(nbr_inputs + 1, nbr_outputs + 1);
	output_units = Matrix(mini_batch_size, nbr_outputs + 1);
	input_units = lower_outputs;
}

void Layer::backprop_up()
{
	forward_pass(*input_units, output_units);
}

void Layer::backprop_down(Layer *upper_layer)
{
	// Calcule les dérivées partielles de l'erreur par rapport au logit et les met dans register_1
	deriv_sumprod(upper_layer);
	// Calcule les dérivées partielles de l'erreur par rapport aux poids et les met dans register_2
	deriv_weights();
}

void Layer::do_top_down(Matrix &features, Matrix &images)
{
	features.fill_left_column(1);
	images.is_multrans_of(features, weights);
	apply_activation(images);
}

cl_float Layer::calculate_cost(const Matrix &label_data, const Matrix &result_data) const
{
	// Ne fait rien, sauf la couche finale (Softmax ou MeanSquare) qui l'override
	return -1;
}

cl_float Layer::calculate_error(const Matrix &label_data, const Matrix &result_data) const
{
	// Ne fait rien, sauf la couche finale (Softmax ou MeanSquare) qui l'override
	return -1;
}

void Layer::display_and_replace(Matrix &input_data)
{
	Matrix output_features;

	displayer->clear_window();
	output_features = Matrix(input_data.get_subrows(), nbr_outputs + 1);
	forward_pass(input_data, output_features);
	displayer->draw_layer(input_data.fetch(), weights.fetch(), output_features.fetch());
	displayer->display_window();
	swap(input_data, output_features);
}

void Layer::apply_activation(Matrix &outputs)
{
	// Ne fait rien, le travail est fait par la classe dérivée
}

void Layer::apply_activation_derivatives(Matrix &outputs)
{
	// Ne fait rien, le travail est fait par la classe dérivée
}

void Layer::deriv_sumprod(Layer *upper_layer)
{
	// Cette fonction est overridée seulement pour la couche finale

	// Prépare la formule récursive : dE/dSj = SOMMEk[dE/dSk(prev) * dSk(prev)/dYj] * dYj/dSj
	// Calcule seulement la somme : SOMMEk[dE/dSk(prev) * Wjk(prev)]
	Matrix &previous_deltas = upper_layer->deltas;
	Matrix &previous_weights = upper_layer->weights;

	deltas.is_multrans_of(previous_deltas, previous_weights);
	// get_register(0).is_multrans_of(previous_layer->get_register(0), previous_layer->weights);
	// get_register(0).display("dE/dYj");
	//  Le résultat doit ensuite être multiplié membre à membre par la dérivée de la fonction d'activation dYj/dSj
	apply_activation_derivatives(output_units);
	// apply_activation_derivatives(output_units);
	// output_units.display("Yj(1-Yj)");
	deltas *= output_units;
	// get_register(0) *= output_units;
	// get_register(0).display("dE/dSIGMAj");
	//  Neutralise pour la suite le calcul des derivées sur la première colonne des neurones
	deltas.fill_left_column(0);
	// get_register(0).fill_left_column(0);
}

void Layer::deriv_weights()
{
	// Applique la formule de la dérivée composée : dE/dWij = dE/dSj * dSj/dWij

	cl_float batch_size = input_units->get_subrows();

	// get_register(1).is_transmult_of(*input_units, get_register(0));
	derivatives.is_transmult_of(*input_units, deltas);
	// Dans le cas où c'est un mini_batch de plusieurs images, calcule la moyenne des dérivées sur le mini-batch
	derivatives /= batch_size;
	// get_register(1) /= (cl_float)optimizer_batch_size;
	//  Pour ne pas utiliser optimizer_batch_size on peut faire :
	// if (input_units->get_subrows() > 1)
	// dE_dWij_reg /= (cl_float)input_units->get_subrows();
}

cl_float Layer::free_energy(const Matrix &input) const
{
	return input.free_nrg(weights);
}