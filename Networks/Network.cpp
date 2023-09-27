//
//  Network.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 13/02/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Network.h"
#include <chrono>

Network::Network(cl_uint inputs)
{
	total_input_units = inputs;
	total_output_units = 0;
	display_mode = MODE_NO_DISPLAY;
	output_units = nullptr;
	input_units = Matrix();
}

Network::~Network()
{
	clear();
}

cl_uint Network::get_type_of_layer(size_t layer_nbr)
{
	return layers[layer_nbr]->get_type();
}

size_t Network::get_nbr_layers()
{
	return layers.size();
}

cl_uint Network::get_nbr_inputs_in_layer(size_t layer_nbr)
{
	return layers[layer_nbr]->get_nbr_inputs();
}

cl_uint Network::get_nbr_outputs_in_layer(size_t layer_nbr)
{
	return layers[layer_nbr]->get_nbr_outputs();
}

Matrix &Network::get_weights_in_layer(size_t layer_nbr)
{
	return layers[layer_nbr]->weights;
}

Matrix &Network::get_derivatives_in_layer(size_t layer_nbr)
{
	return layers[layer_nbr]->derivatives;
}

Matrix Network::get_outputs_in_layer(size_t layer_nbr, const Matrix &input_data)
{
	Matrix current_input = input_data;
	Matrix output_results;

	for (size_t i = 0; i < layer_nbr + 1; i++)
	{
		output_results = Matrix(current_input.get_rows(), get_nbr_outputs_in_layer(i) + 1);
		layers[i]->forward_pass(current_input, output_results);
		output_results.fill_left_column(1);
		current_input = output_results;
	}

	return output_results;
}

void Network::set_display(int mode)
{
	display_mode = mode;
}

void Network::add_layer(int layer_type, cl_uint nbr_outputs, cl_float param)
{
	cl_uint nbr_inputs;
	Layer *new_layer;
	string window_title;
	int window_mode;

	// Create a new Sigmoid
	if (layers.size() == 0)
		nbr_inputs = total_input_units;
	else
		nbr_inputs = total_output_units;

	switch (layer_type)
	{
	case LAYER_SIGMOID:
		new_layer = new Sigmoid(nbr_inputs, nbr_outputs);
		break;
	case LAYER_RELU:
		new_layer = new ReLU(nbr_inputs, nbr_outputs);
		break;
	case LAYER_DROPOUT:
		if (nbr_outputs != nbr_inputs)
			throw std::runtime_error("Network - Dropout must have equal number of inputs and outputs");
		new_layer = new Dropout(nbr_inputs, param);
		break;
	case LAYER_SOFTMAX:
		new_layer = new Softmax(nbr_inputs, nbr_outputs);
		break;
	case LAYER_MEANSQR:
		new_layer = new MeanSquare(nbr_inputs, nbr_outputs);
		break;
	default:
		throw std::runtime_error("Network - Unknown layer type");
		break;
	}

	total_output_units = nbr_outputs;

	// Sets up the displayer for that layer

	window_title = new_layer->get_name() + "[" + std::to_string(layers.size()) + "]";
	window_title += " " + std::to_string(nbr_inputs) + "x" + std::to_string(nbr_outputs);
	if (display_mode == MODE_NO_DISPLAY)
	{
		new_layer->displayer = nullptr;
	}
	else
	{
		window_mode = display_mode;
		if (display_mode == MODE_MIXED)
		{
			window_mode = MODE_INPUTS_AND_WEIGHTS;
			if ((layer_type == LAYER_SOFTMAX) || (layer_type == LAYER_MEANSQR))
			{
				window_mode = MODE_INPUTS_AND_OUTPUTS;
			}
		}
		new_layer->displayer = new Displayer(window_title, nbr_inputs, nbr_outputs, window_mode);
	}
	logs.add(window_title);

	// Pushes the new layer onto the Network layers
	layers.push_back(new_layer);
}

void Network::clear()
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		delete layers[i];
	}
	layers.clear();
}

void Network::setup(cl_uint batch_size)
{
	Matrix *lower_outputs = &input_units;

	for (size_t i = 0; i < layers.size(); i++)
	{
		layers[i]->setup(batch_size, lower_outputs);
		lower_outputs = &layers[i]->output_units;
	}
	output_units = lower_outputs;
}

void Network::forward_pass(Matrix &input_data,
													 Matrix &output_results)
{
	Matrix current_input = input_data;

	for (size_t i = 0; i < layers.size(); i++)
	{
		output_results = Matrix(current_input.get_rows(), get_nbr_outputs_in_layer(i) + 1);
		layers[i]->forward_pass(current_input, output_results);
		output_results.fill_left_column(1);
		current_input = output_results;
	}
}

void Network::backprop(const Matrix &inputs,
											 const Matrix &labels)
{
	Layer *upper_layer = nullptr;

	input_units.is_shallow_copy_of(inputs);
	for (size_t i = 0; i < layers.size(); i++)
	{
		layers[i]->backprop_up();
	}
	*output_units -= labels;
	for (size_t i = layers.size(); i-- != 0;)
	{
		layers[i]->backprop_down(upper_layer);
		upper_layer = layers[i];
	}
}

cl_float Network::calculate_cost(const Matrix &label_data,
																 const Matrix &result_data)
{
	cl_float cost;
	cost = layers.back()->calculate_cost(label_data, result_data);
	return cost;
}

cl_float Network::calculate_error(const Matrix &label_data,
																	const Matrix &result_data)
{
	cl_float error;
	error = layers.back()->calculate_error(label_data, result_data);
	return error;
}

Matrix Network::reorder(Matrix &input_data,
												Matrix &label_data)
{

	Matrix output_results;
	DataBlock original_inputs;
	DataBlock original_labels;
	DataBlock ordered_images(100, 785);
	Matrix ordered_samples;
	unsigned int img_index = 0;
	unsigned int sub_orders[10] = {0};
	unsigned int num_finished = 0;

	input_data.select_subset(0, 1000);
	original_inputs = input_data.fetch();
	label_data.select_subset(0, 1000);
	original_labels = label_data.fetch();

	while (num_finished < 10)
	{
		for (unsigned int i = 0; i < 10; i++)
		{
			if (original_labels.cpu_buffer[img_index * 11 + i + 1] == 1)
			{
				const unsigned int col = i;
				const unsigned int row = sub_orders[col];
				if (row < 10)
				{
					const unsigned int new_index = col * 10 + row;
					for (unsigned int j = 0; j < 785; j++)
					{
						ordered_images.cpu_buffer[new_index * 785 + j] = original_inputs.cpu_buffer[img_index * 785 + j];
					}
					sub_orders[col]++;
					if (sub_orders[col] == 10)
						num_finished++;
				}
			}
		}
		img_index++;
	}

	ordered_samples = Matrix(ordered_images);
	return ordered_samples;
}

void Network::display(const Matrix &input_data)
{
	Layer *cur_layer;
	Matrix display_data = input_data;

	for (unsigned int i = 0; i < layers.size(); i++)
	{
		cur_layer = layers[i];
		cur_layer->display_and_replace(display_data);
	}
}

void Network::save(string filename)
{
	string filepath = "Backups/" + filename;
	size_t nbr_layers;
	cl_uint layer_type;
	cl_uint nbr_outputs;
	float param;

	ofstream file(filepath + ".dln", ios::out | ios::binary | ios::trunc);
	// ios::trunc	If the file is opened for output operations and it already existed, its previous content is deleted and replaced by the new one.
	if (!file.is_open())
		throw std::runtime_error("Unable to open file");

	nbr_layers = get_nbr_layers();
	file.write((char *)&nbr_layers, sizeof(nbr_layers));
	file.write((char *)&total_input_units, sizeof(total_input_units));

	for (size_t i = 0; i < get_nbr_layers(); i++)
	{
		layer_type = get_type_of_layer(i);
		file.write((char *)&layer_type, sizeof(layer_type));
		nbr_outputs = get_nbr_outputs_in_layer(i);
		file.write((char *)&nbr_outputs, sizeof(nbr_outputs));
		param = 0;
		if (layer_type == LAYER_DROPOUT)
		{
			Dropout *dropout_layer = dynamic_cast<Dropout *>(layers[i]);
			param = dropout_layer->get_retain_rate();
		}
		file.write((char *)&param, sizeof(param));
		Extractor::save_matrix(filepath + "_layer" + std::to_string(i) + ".wgt", get_weights_in_layer(i));
	}

	file.close();
}

void Network::load(string filename)
{
	string filepath = "Backups/" + filename;
	size_t nbr_layers;
	cl_uint layer_type;
	cl_uint nbr_outputs = 0;
	cl_float param;

	ifstream file(filepath + ".dln", ios::in | ios::binary | ios::ate);
	if (!file.is_open())
		throw "Unable to open file";

	clear();

	file.seekg(0, ios::beg);
	file.read((char *)&nbr_layers, sizeof(nbr_layers));
	file.read((char *)&total_input_units, sizeof(total_input_units));

	for (size_t i = 0; i < nbr_layers; i++)
	{
		file.read((char *)&layer_type, sizeof(layer_type));
		file.read((char *)&nbr_outputs, sizeof(nbr_outputs));
		file.read((char *)&param, sizeof(param));

		add_layer(layer_type, nbr_outputs, param);
		get_weights_in_layer(i) = Extractor::load_matrix(filepath + "_layer" + std::to_string(i) + ".wgt");
	}
	output_units = nullptr;
	// input_units = Matrix();
	total_output_units = nbr_outputs;
}
