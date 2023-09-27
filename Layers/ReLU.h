//
//  ReLU.h
//  deep learning
//
//  Created by Sebastien Raffray on 30/03/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__ReLU__
#define __deep_learning__ReLU__

#include "Layer.h"

class ReLU : public Layer
{
public:

	ReLU();
	ReLU(const cl_uint nbr_input_units, const cl_uint nbr_output_units);
	virtual ~ReLU();

private:

	virtual void		apply_activation(Matrix &outputs);
	virtual void		apply_activation_derivatives(Matrix &outputs);
	};

#endif /* defined(__deep_learning__ReLU__) */
