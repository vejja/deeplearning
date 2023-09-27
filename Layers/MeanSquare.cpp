//
//  MeanSquare.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 22/06/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "MeanSquare.h"

MeanSquare::MeanSquare() : Layer()
{
}

MeanSquare::MeanSquare(const cl_uint nbr_visible_units, const cl_uint nbr_hidden_units) : Layer(nbr_visible_units, nbr_hidden_units)
{
	layer_type = LAYER_MEANSQR;
	layer_name = "MeanSquare";
}

MeanSquare::~MeanSquare()
{
}

void MeanSquare::apply_activation(Matrix &outputs)
{
	//  Nothing to do : this is the id function
	//  zk = Sk
}

cl_float MeanSquare::calculate_cost(const Matrix &label_data, const Matrix &result_data) const
{
	// Mean-square error function is used as cost function for MeanSquare Layer
	// 1/2 SUMj (yj - tj)^2
	// However for convenience purpose, here we return the euclidian distance which is
	// 1/N sqrt [ SUMj (yj - tj)^2 ]
	Matrix difference = label_data;
	difference -= result_data;
	return difference.norm() / difference.get_rows();
}

cl_float MeanSquare::calculate_error(const Matrix &label_data, const Matrix &result_data) const
{
	// Ca n'a pas tellement de sens ici
	// Parce que result n'est pas la prediction d'appartenir a une classe donnée
	// Ici du coup on suit l'erreur maximale sur l'échantillon
	Matrix difference = label_data;
	difference -= result_data;
	// difference.display("difference pour adri");
	return difference.absmax();
}

void MeanSquare::deriv_sumprod(Layer *previous_layer)
{
	// C'est la même formule que pour softmax + cross entropy
	// Override la fonction de base pour la derniere couche
	// output_units doit etre pre-loade avec output_units - labels

	// Ignore la couche precedente (il n'y en a pas) et initialise les deltas
	deltas.is_deep_copy_of(output_units);
	// Neutralise pour la suite le calcul des derivées sur la première colonne des neurones
	deltas.fill_left_column(0);
}
