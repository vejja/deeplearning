//
//  Optimizer.h
//  deep learning
//
//  Created by Sebastien Raffray on 04/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Optimizer__
#define __deep_learning__Optimizer__

#include "../Networks.h"
#include "../Utilities/Logger.h"

class Optimizer
{
public:
	std::string		optimizer_name;

	Optimizer();
	virtual ~Optimizer();
	
	virtual void	initialise(cl_uint mini_batch_size,
							   Network &network);
	void			learn(Matrix &train_images,
						  Matrix &train_labels,
						  Matrix &test_images,
						  Matrix &test_labels,
						  cl_uint max_epochs);
	
	void			autoencode(Matrix &data,
							   cl_uint max_epochs);
	
	virtual void	optimize(Matrix &inputs, Matrix &outputs) = 0;
	
protected:
	Network*		m_network;
	cl_uint			m_batch_size;
	
	// Helper functions
	void			add_log(std::string logstring);
	Matrix			reorder(Matrix &inputs, Matrix &labels);
	void			display(Matrix &inputs);
	void			forward_pass(Matrix &inputs, Matrix &outputs);
	cl_float		calculate_cost(Matrix &labels, Matrix &results);
	cl_float		calculate_error(Matrix &labels, Matrix &results);
	size_t			get_nbr_layers();
	cl_uint			get_nbr_inputs_in_layer(size_t layer_nbr);
	cl_uint			get_nbr_outputs_in_layer(size_t layer_nbr);
	Matrix&			get_weights_in_layer(size_t layer_nbr);
	Matrix&			get_derivatives_in_layer(size_t layer_nbr);
};

#endif /* defined(__deep_learning__Optimizer__) */
