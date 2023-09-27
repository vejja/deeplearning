//
//  MeanSquare.h
//  deep learning
//
//  Created by Sebastien Raffray on 22/06/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__MeanSquare__
#define __deep_learning__MeanSquare__


#include "Layer.h"

class MeanSquare : public Layer
{
public:
	
	MeanSquare();
	MeanSquare(const cl_uint nbr_visible_units, const cl_uint nbr_hidden_units);
	virtual ~MeanSquare();
	
	virtual cl_float	calculate_cost(const Matrix &label_data, const Matrix &result_data) const;
	virtual cl_float	calculate_error(const Matrix &label_data, const Matrix &result_data) const;
	
private:
	
	virtual void		deriv_sumprod(Layer *previous_layer);
	virtual void		apply_activation(Matrix &outputs);
};


#endif /* defined(__deep_learning__MeanSquare__) */


