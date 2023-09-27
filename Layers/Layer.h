//
//  Layer.h
//  deep learning
//
//  Created by Sébastien Raffray on 03/03/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Layer__
#define __deep_learning__Layer__

#include "../GPU Acceleration/Matrix.h"
#include "../Utilities/Displayer.h"
#include <string>

#define LAYER_UNDEFINED 0
#define LAYER_SIGMOID 1
#define LAYER_RELU 2
#define LAYER_DROPOUT 3
#define LAYER_SOFTMAX 4
#define LAYER_MEANSQR 5

class Layer
{

public:
	Matrix weights;
	Displayer *displayer;
	Matrix deltas;
	Matrix derivatives;
	Matrix output_units;
	Matrix *input_units;

	Layer();
	Layer(cl_uint nbr_input_units,
				cl_uint nbr_output_units,
				cl_float mean = 0.0f,
				cl_float stdev = 0.01f);
	virtual ~Layer();

	cl_uint get_type() const;
	string get_name() const;
	cl_uint get_nbr_inputs() const;
	cl_uint get_nbr_outputs() const;
	string get_title() const;
	cl_uint get_batch_size() const;

	void init_weights(const Matrix &input_images);
	void generative_training(Matrix &input_images, const cl_uint nbr_epochs);
	virtual void forward_pass(Matrix &inputs, Matrix &outputs);
	virtual void setup(cl_uint mini_batch_size,
										 Matrix *lower_outputs);
	virtual void backprop_up();
	virtual void backprop_down(Layer *upper_layer);
	virtual cl_float calculate_cost(const Matrix &label_data, const Matrix &result_data) const;
	virtual cl_float calculate_error(const Matrix &label_data, const Matrix &result_data) const;

	void display_and_replace(Matrix &input_data);

protected:
	cl_uint layer_type;
	string layer_name;
	cl_uint nbr_inputs;
	cl_uint nbr_outputs;

private:
	virtual void do_top_down(Matrix &features, Matrix &images);
	virtual void apply_activation(Matrix &outputs);
	virtual void apply_activation_derivatives(Matrix &outputs);
	virtual void deriv_sumprod(Layer *previous_layer);
	void deriv_weights();
	cl_float free_energy(const Matrix &input) const;
};

#endif /* defined(__deep_learning__Layer__) */
