//
//  Dropout.h
//  deep learning
//
//  Created by Sebastien Raffray on 02/04/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Dropout__
#define __deep_learning__Dropout__

#include "Layer.h"

class Dropout : public Layer
{
public:
	Dropout();
	Dropout(cl_uint nbr_input_and_output_units, cl_float init_retain_rate = 0.5f);
	virtual ~Dropout();

	cl_float get_retain_rate() const;
	virtual void forward_pass(Matrix &inputs, Matrix &outputs);
	virtual void setup(cl_uint mini_batch_size,
										 Matrix *lower_outputs);
	virtual void backprop_up();
	virtual void backprop_down(Layer *upper_layer);

private:
	cl_float retain_rate;
	Matrix mask;
};

#endif /* defined(__deep_learning__Dropout__) */