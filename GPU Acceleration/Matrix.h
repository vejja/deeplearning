//
//  Matrix.h
//  deep learning
//
//  Created by Sébastien Raffrayay on 12/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Matrix__
#define __deep_learning__Matrix__

#include "GPU.h"
#include "../Utilities/DataBlock.h"
#include <chrono>
#include <string>

class Matrix
{

public:
	Matrix();															 // default constructor, empty matrix
	Matrix(cl_uint rows, cl_uint columns); // overloaded constructor, allocates GPU memory
	Matrix(DataBlock &input);							 // constructor, fills GPU memory from DataBlock
	Matrix(const Matrix &source);					 // copy constructor
	Matrix(Matrix &&matrix);							 // move constructor
	~Matrix();														 // destructor

	Matrix &operator=(Matrix matrix);								 // assignment operator
	friend void swap(Matrix &first, Matrix &second); // copy-and-swap idiom implementation

	size_t nb_elements() const;
	size_t nb_subelements() const;
	size_t start_offset() const;
	cl_uint get_rows() const;
	cl_uint get_cols() const;
	cl_uint get_subrows() const;

	Matrix &operator=(cl_float rhs_scalar);				// fill with scalar
	Matrix &operator+=(const Matrix &rhs_matrix); // in-place addition
	Matrix &operator+=(cl_float rhs_scalar);			// in-place memberwise addition
	Matrix &operator-=(const Matrix &rhs_matrix); // in-place substraction
	Matrix &operator-=(cl_float rhs_scalar);			// in-place memberwise substraction
	Matrix &operator*=(cl_float rhs_scalar);			// in-place scalar product
	Matrix &operator*=(const Matrix &rhs_matrix); // in-place memberwise product
	Matrix &operator/=(cl_float rhs_scalar);			// in-place scalar division
	bool operator==(const Matrix &other) const;		// comparison of matrix dimensions
	bool operator!=(const Matrix &other) const;		// comparison of matrix dimensions

	void load_from_cpu(DataBlock &input);
	DataBlock fetch() const;
	DataBlock fetch(cl_uint start_row, cl_uint nbr_rows);

	cl_float norm() const;	 // distance as euclidian norm
	cl_float absmax() const; // max absolute value of all elements

	void is_shallow_copy_of(const Matrix &matrix);
	void is_deep_copy_of(const Matrix &matrix);
	void is_transp_of(const Matrix &matrix);
	void select_subset(cl_uint start, cl_uint nbr);
	void select_full_set();
	void add_scaled(const Matrix &input, cl_float scalar);
	void is_mult_of(const Matrix &first, const Matrix &second);
	void add_mult_of(const Matrix &first, const Matrix &second);
	void sub_mult_of(const Matrix &first, const Matrix &second);
	void is_transmult_of(const Matrix &first, const Matrix &second);
	void is_multrans_of(const Matrix &first, const Matrix &second);
	void add_transmult_of(const Matrix &first, const Matrix &second);
	void sub_transmult_of(const Matrix &first, const Matrix &second);

	void fill_right_block(const Matrix &matrix);
	void fill_left_column(cl_float value);
	void fill_with_random_uniform(cl_float lo, cl_float hi);
	void fill_with_random_gaussian(cl_float mean, cl_float stdev);
	void fill_with_random_binary(cl_float prob);
	void fill_top_row(cl_float value);
	void left_is_avg_pxls(const Matrix &images);
	void apply_sampling(const Matrix &random_samples);
	void apply_sigmoid();
	void apply_relu();
	void apply_softmax();
	void apply_square();
	void apply_sigmoid_deriv();
	void apply_relu_deriv();
	void is_sqtrmult_of(const Matrix &left, const Matrix &top);
	void update_rolling_average(const Matrix &new_data, const Matrix &timings);
	void update_learning_rates(const Matrix &mean_gradient, const Matrix &var_gradient, const Matrix &est_hessian);
	void update_timings(const Matrix &mean_gradient, const Matrix &var_gradient);
	void apply_mean_square_update(cl_float decay_rate, const Matrix &derivatives);
	void add_with_rms_scaling(const Matrix &derivatives, cl_float epsilon, const Matrix &mean_squares, cl_float smoothing_factor);
	void apply_rms_scaling(const Matrix &learning_rates, const Matrix &mean_squares, cl_float damping_factor);
	void adapt_and_clip(const Matrix &gradients, const Matrix &velocities, cl_float adapt_rate, cl_float min_rate, cl_float max_rate);
	cl_float free_nrg(const Matrix &neurons) const;
	static cl_float cross_entropy(const Matrix &goals, const Matrix &values);
	static cl_float error_rate(const Matrix &first, const Matrix &second);
	void shuffle_rows();
	void standardize();

	void display(std::string name) const;

private:
	static GPU gpu;
	static cl_uint total_mem_alloc;

	cl_mem gpu_buffer;
	cl_uint nb_rows;
	cl_uint nb_columns;
	cl_uint start_subrow;
	cl_uint nb_subrows;

	void create();
	void release();
	void scale_by(cl_float scalar);
	void apply_memberprod(const Matrix &other);
	void apply_memberadd(cl_float scalar);
	void fill_with_float(cl_float value);

	void scale_then_add_multiplied_scaled(cl_float self_scalar, const Matrix &X, const Matrix &Y, cl_float mult_scalar);
	void scale_then_add_transmult_scaled(cl_float self_scalar, const Matrix &first, const Matrix &nofirst, cl_float mult_scalar);
	void scale_then_add_multrans_scaled(cl_float self_scalar, const Matrix &first, const Matrix &nofirst, cl_float mult_scalar);
};

#endif /* defined(__deep_learning__Matrix__) */
