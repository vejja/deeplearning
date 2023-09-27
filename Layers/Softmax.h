//
//  Softmax.h
//  deep learning
//
//  Created by Sebastien Raffray on 13/02/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Softmax__
#define __deep_learning__Softmax__

#include "Layer.h"

class Softmax : public Layer
{
public:

    Softmax();
    Softmax(const cl_uint nbr_visible_units, const cl_uint nbr_hidden_units);
    virtual ~Softmax();

	virtual cl_float	calculate_cost(const Matrix &label_data, const Matrix &result_data) const;
	virtual cl_float	calculate_error(const Matrix &label_data, const Matrix &result_data) const;

private:

	virtual void		deriv_sumprod(Layer *previous_layer);
	virtual void		apply_activation(Matrix &outputs);
};

#endif /* defined(__deep_learning__Softmax__) */