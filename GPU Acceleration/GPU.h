//
//  GPU.h
//  deep learning
//
//  Created by Sébastien Raffray on 11/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__GPU__
#define __deep_learning__GPU__

#include <iostream>
#include <fstream>
#include <math.h>
#include <clBLAS.h>

class GPU
{
public:
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue command_queue;
	cl_kernel sigmoid_kernel;
	cl_kernel relu_kernel;
	cl_kernel softmax_kernel;
	cl_kernel square_kernel;
	cl_kernel exponential_kernel;
	cl_kernel sum_reduce_kernel;
	cl_kernel divide_kernel;
	cl_kernel sigmoid_deriv_kernel;
	cl_kernel relu_deriv_kernel;
	cl_kernel member_prod_kernel;
	cl_kernel member_add_kernel;
	cl_kernel sqtr_prod_kernel;
	cl_kernel rolling_avg_kernel;
	cl_kernel update_rates_kernel;
	cl_kernel update_timings_kernel;
	cl_kernel mean_square_update_kernel;
	cl_kernel add_rms_scale_kernel;
	cl_kernel rms_scale_kernel;
	cl_kernel adapt_clip_kernel;
	cl_kernel cross_entropy_kernel;
	cl_kernel error_rate_kernel;
	cl_kernel fill_kernel;
	cl_kernel fill_left_kernel;
	cl_kernel fill_top_kernel;
	cl_kernel avg_pix_kernel;
	cl_kernel fill_right_kernel;
	cl_kernel binary_distrib_kernel;
	cl_kernel transpose_kernel;
	cl_kernel free_nrg_1_kernel;
	cl_kernel free_nrg_2_kernel;
	cl_kernel shuffle_rows_kernel;
	cl_kernel standardize_kernel;

	GPU();
	~GPU();
	cl_int getPlatformID(cl_platform_id &platform);
	void displayLogProgram(cl_program program);
	void customKernelsSetup();
	void customKernelsRelease();
};

#endif /* defined(__deep_learning__GPU__) */
