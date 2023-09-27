//
//  Adam.h
//  deep learning
//
//  Created by Sebastien Raffray on 15/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Adam__
#define __deep_learning__Adam__

#include "Optimizer.h"

class Adam : public Optimizer {
public:
	Adam(cl_float init_learning_rate = 0.001,
		 cl_float init_gradient_decay = 0.9,
		 cl_float init_squares_decay = 0.999,
		 cl_float damping_factor = 1e-08,
		 cl_float decay_scaling = 1 - 1e-08);
	virtual ~Adam();
	
	Adam&		with_init_learning_rate(cl_float init_learning_rate);
	Adam&		with_init_gradient_decay(cl_float init_gradient_decay);
	Adam&		with_init_squares_decay(cl_float init_squares_decay);
	Adam&		with_damping_factor(cl_float damping_factor);
	Adam&		with_decay_scaling(cl_float decay_scaling);

	virtual void		initialise(cl_uint mini_batch_size,
								   Network &network);
	virtual void		optimize(Matrix &inputs,
								 Matrix &outputs);
	
private:
	cl_uint			nbr_iter;				// t = Number of iterations, starting at 0.
	cl_float		alpha;					// alpha = Learning rate
	cl_float		beta1;					// Exponential decay for the biased gradient moving average
	cl_float		beta2;					// Exponential decay for the squared gradients moving average
	cl_float		epsilon;				// epsilon = Safety offset for division by estimate of second moment.
	cl_float		lambda;					// lambda = Used to decay the beta1 momentum
	cl_float		beta1_t;				// beta(1,t) = beta1, but scaled with lambda
	vector<Matrix*>	biased_avg_gradients;	// m(t) = Biased estimate of first moment.
	vector<Matrix*>	biased_avg_squares;		// v(t) = Biased estimate of second moment.


};

#endif /* defined(__deep_learning__Adam__) */
