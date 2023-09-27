//
//  Optimizer.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 04/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "Optimizer.h"

Optimizer::Optimizer()
{
	optimizer_name = "None";
}


Optimizer::~Optimizer()
{
}


void		Optimizer::initialise(cl_uint mini_batch_size,
								  Network &network)
{
	m_batch_size = mini_batch_size;
	m_network = &network;
	
	m_network->setup(mini_batch_size);

	add_log("Optimizer : " + optimizer_name);
	add_log("Batch size " + std::to_string(mini_batch_size));
}

void		Optimizer::learn(Matrix &train_images,
							 Matrix &train_labels,
							 Matrix &test_images,
							 Matrix &test_labels,
							 cl_uint max_epochs)
{
	cl_uint max_img;
	cl_float cost;
	cl_float error;
	cl_uint epoch = 0;
	Matrix results;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
	std::chrono::duration<double> elapsed_seconds;
	
	max_img = (train_images.get_rows() / m_batch_size) * m_batch_size;
	
	while ((epoch < max_epochs) && (!Displayer::exit_requested))
	{
		start_time = std::chrono::high_resolution_clock::now();
		add_log("*** Learning epoch #" + std::to_string(epoch));

		for (cl_uint cur_img = 0; cur_img < max_img; cur_img += m_batch_size)
		{
			train_images.select_subset(cur_img, m_batch_size);
			train_labels.select_subset(cur_img, m_batch_size);
			optimize(train_images, train_labels);
		}
		
		train_images.select_full_set();
		train_labels.select_full_set();
		forward_pass(train_images, results);
		cost = calculate_cost(train_labels, results);
		error = calculate_error(train_labels, results);
		add_log ("On Training Set - Cost : " + std::to_string(cost)
				  + " - Error : " +std::to_string(error * 100) + "%");
		
		forward_pass(test_images, results);
		cost = calculate_cost(test_labels, results);
		error = calculate_error(test_labels, results);
		add_log("On Validation Set - Cost : " + std::to_string(cost)
				 + " - Error : " + std::to_string(error * 100) + "%");
		
		end_time = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end_time - start_time;
		add_log("Finished epoch #" + std::to_string(epoch) + " in "
				 + std::to_string(elapsed_seconds.count()) + " seconds.");
		
		// Displays the Network
		train_images.select_subset(0, 100);
		Matrix reordered_inputs;
		reordered_inputs = reorder(train_images, train_labels);
		display(reordered_inputs);
		train_images.select_full_set();
		epoch++;
	}
	std::cout << std::endl;
}

void		Optimizer::autoencode(Matrix &data,
								  cl_uint max_epochs)
{
	cl_uint max_img;
	cl_float cost;
	cl_float error;
	cl_uint epoch = 0;
	Matrix results;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
	std::chrono::duration<double> elapsed_seconds;
	
	max_img = (data.get_rows() / m_batch_size) * m_batch_size;
	
	while ((epoch < max_epochs) && (!Displayer::exit_requested))
	{
		start_time = std::chrono::high_resolution_clock::now();
		add_log("*** Learning epoch #" + std::to_string(epoch));
		data.shuffle_rows();
		
		for (cl_uint cur_img = 0; cur_img < max_img; cur_img += m_batch_size)
		{
			data.select_subset(cur_img, m_batch_size);
			optimize(data, data);
		}
		
		if (epoch % 20 == 0) {
			data.select_full_set();
			forward_pass(data, results);
			cost = calculate_cost(data, results);
			error = calculate_error(data, results);
			add_log ("On Self Set - Cost : " + std::to_string(cost)
				 + " - Error : " +std::to_string(error * 100) + "%");
			
			// Displays the Network
			data.select_subset(0, 100);
			display(data);
		}
		
		data.select_full_set();
		end_time = std::chrono::high_resolution_clock::now();
		elapsed_seconds = end_time - start_time;
		add_log("Finished epoch #" + std::to_string(epoch) + " in "
				+ std::to_string(elapsed_seconds.count()) + " seconds.");
		epoch++;
	}
	std::cout << std::endl;
}

void		Optimizer::add_log(std::string logstring)
{
	m_network->logs.add(logstring);
}

Matrix		Optimizer::reorder(Matrix &inputs, Matrix &labels)
{
	return m_network->reorder(inputs, labels);
}

void		Optimizer::display(Matrix &inputs)
{
	m_network->display(inputs);
}

void		Optimizer::forward_pass(Matrix &inputs, Matrix &outputs)
{
	m_network->forward_pass(inputs, outputs);
}

cl_float	Optimizer::calculate_cost(Matrix &labels, Matrix &results)
{
	return m_network->calculate_cost(labels, results);
}

cl_float	Optimizer::calculate_error(Matrix &labels, Matrix &results)
{
	return m_network->calculate_error(labels, results);
}

size_t		Optimizer::get_nbr_layers()
{
	return m_network->get_nbr_layers();
}

cl_uint		Optimizer::get_nbr_inputs_in_layer(size_t layer_nbr)
{
	return m_network->get_nbr_inputs_in_layer(layer_nbr);
}

cl_uint		Optimizer::get_nbr_outputs_in_layer(size_t layer_nbr)
{
	return m_network->get_nbr_outputs_in_layer(layer_nbr);
}

Matrix&		Optimizer::get_weights_in_layer(size_t layer_nbr)
{
	return m_network->get_weights_in_layer(layer_nbr);
}

Matrix&		Optimizer::get_derivatives_in_layer(size_t layer_nbr)
{
	return m_network->get_derivatives_in_layer(layer_nbr);
}