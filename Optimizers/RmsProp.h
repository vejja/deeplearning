//
//  RmsProp.h
//  deep learning
//
//  Created by Sebastien Raffray on 04/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__RmsProp__
#define __deep_learning__RmsProp__

#include "Optimizer.h"

class RmsProp : public Optimizer {
public:
	RmsProp(cl_float learning_rate = 0.001f,
			cl_float decay_rate = 0.9f,
			cl_float damping_factor = 0.01f);
	virtual ~RmsProp();
	
	virtual void		initialise(cl_uint mini_batch_size,
								   Network &network);
	virtual void		optimize(Matrix &inputs,
								 Matrix &outputs);
	
private:
	vector<Matrix*>	m_mean_squares;
	cl_float		m_learning_rate;
	cl_float		m_decay_rate;
	cl_float		m_damping_factor;
};


#endif /* defined(__deep_learning__RmsProp__) */
