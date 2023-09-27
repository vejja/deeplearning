//
//  RMS-Nesterov.h
//  deep learning
//
//  Created by Sebastien Raffray on 15/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__RmsNesterov__
#define __deep_learning__RmsNesterov__

#include "Optimizer.h"

class RmsNesterov : public Optimizer {
public:
	RmsNesterov(cl_float start_learning_rate = 0.001f,
				cl_float decay_rate = 0.9f,
				cl_float damping_factor = 0.01f,
				cl_float momentum = 0.0f,
				cl_float min_learning_rate = 1e-06,
				cl_float max_learning_rate = 1e+02,
				cl_float rate_adapt = 0.0f);
	virtual ~RmsNesterov();
	
	RmsNesterov&		with_start_learning_rate(cl_float start_learning_rate);
	RmsNesterov&		with_decay_rate(cl_float decay_rate);
	RmsNesterov&		with_damping_factor(cl_float damping_factor);
	RmsNesterov&		with_momentum(cl_float momentum);
	RmsNesterov&		with_min_learning_rate(cl_float min_learning_rate);
	RmsNesterov&		with_max_learning_rate(cl_float max_learning_rate);
	RmsNesterov&		with_rate_adapt(cl_float rate_adapt);
	
	virtual void		initialise(cl_uint mini_batch_size,
								   Network &network);
	virtual void		optimize(Matrix &inputs,
								 Matrix &outputs);
	
private:
	vector<Matrix*>	m_mean_squares;
	vector<Matrix*> m_velocity;
	vector<Matrix*> m_learning_rates;
	cl_float		m_start_learning_rate;
	cl_float		m_decay_rate;
	cl_float		m_damping_factor;
	cl_float		m_momentum;
	cl_float		m_min_learning_rate;
	cl_float		m_max_learning_rate;
	cl_float		m_rate_adapt;
};


#endif /* defined(__deep_learning__RmsNesterov__) */
