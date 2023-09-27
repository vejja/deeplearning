//
//  Sigmoid.h
//  deep learning
//
//  Created by Sébastien Raffray on 09/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Sigmoid__
#define __deep_learning__Sigmoid__

#include "Layer.h"

class Sigmoid : public Layer
{
public:

	Sigmoid();
	Sigmoid(const cl_uint nbr_input_units, const cl_uint nbr_output_units);
    virtual ~Sigmoid();

private:

	virtual void		apply_activation(Matrix &outputs);
	virtual void		apply_activation_derivatives(Matrix &outputs);

};

#endif /* defined(__deep_learning__Sigmoid__) */

