//
//  Matrix.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 12/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Matrix.h"

GPU Matrix::gpu;
cl_uint Matrix::total_mem_alloc = 0;

Matrix::Matrix() // default constructor
{
	gpu_buffer = nullptr;
	nb_rows = 0;
	nb_columns = 0;
	start_subrow = 0;
	nb_subrows = 0;
}

Matrix::Matrix(cl_uint rows, cl_uint columns) : Matrix() // overloaded constructor, allocates GPU memory
{
	nb_rows = rows;
	nb_columns = columns;
	nb_subrows = nb_rows;
	create();
}

Matrix::Matrix(DataBlock &input) : Matrix(input.nb_rows, input.nb_cols) // overloaded constructor, fills GPU memory from Image extraction source
{
	load_from_cpu(input);
}

Matrix::Matrix(const Matrix &source) : Matrix(source.nb_subrows, source.nb_columns) // copy constructor
{
	if (source.gpu_buffer)
		is_deep_copy_of(source);
}

Matrix::Matrix(Matrix &&matrix) : Matrix() // move constructor
{
	swap(*this, matrix);
}

Matrix::~Matrix() // destructor
{
	release();
}

Matrix &Matrix::operator=(Matrix matrix) // assignment operator makes a copy if value is passed, makes a move if rhs-exp is passed
{
	swap(*this, matrix);
	return *this;
}

void swap(Matrix &first, Matrix &second) // required for copy-and-swap idiom implementation
{
	swap(first.gpu_buffer, second.gpu_buffer);
	swap(first.nb_rows, second.nb_rows);
	swap(first.nb_columns, second.nb_columns);
	swap(first.start_subrow, second.start_subrow);
	swap(first.nb_subrows, second.nb_subrows);
}

size_t Matrix::nb_elements() const
{
	return (nb_rows * nb_columns);
}

size_t Matrix::nb_subelements() const
{
	return (nb_subrows * nb_columns);
}

size_t Matrix::start_offset() const
{
	return (start_subrow * nb_columns);
}

cl_uint Matrix::get_rows() const
{
	return nb_rows;
}

cl_uint Matrix::get_cols() const
{
	return nb_columns;
}

cl_uint Matrix::get_subrows() const
{
	return nb_subrows;
}

Matrix &Matrix::operator=(cl_float rhs_scalar)
{
	fill_with_float(rhs_scalar);
	return *this;
}

Matrix &Matrix::operator+=(const Matrix &rhs_matrix) // in-place addition
{
	add_scaled(rhs_matrix, 1);
	return *this;
}

Matrix &Matrix::operator+=(cl_float rhs_scalar)
{
	apply_memberadd(rhs_scalar);
	return *this;
}

Matrix &Matrix::operator-=(const Matrix &rhs_matrix) // in-place substraction
{
	add_scaled(rhs_matrix, -1);
	return *this;
}

Matrix &Matrix::operator-=(cl_float rhs_scalar)
{
	apply_memberadd(-rhs_scalar);
	return *this;
}

Matrix &Matrix::operator*=(cl_float rhs_scalar) // in-place scalar product
{
	scale_by(rhs_scalar);
	return *this;
}

Matrix &Matrix::operator*=(const Matrix &rhs_matrix) // in-place memberwise product
{
	apply_memberprod(rhs_matrix);
	return *this;
}

Matrix &Matrix::operator/=(cl_float rhs_scalar) // in-place scalar division
{
	scale_by(1 / rhs_scalar);
	return *this;
}

bool Matrix::operator==(const Matrix &other) const // comparison of matrix dimensions
{
	return ((nb_subrows == other.nb_subrows) && (nb_columns == other.nb_columns));
}

bool Matrix::operator!=(const Matrix &other) const // comparison of matrix dimensions
{
	return !(*this == other);
}

void Matrix::is_shallow_copy_of(const Matrix &matrix)
{
	nb_rows = matrix.nb_rows;
	nb_columns = matrix.nb_columns;
	start_subrow = matrix.start_subrow;
	nb_subrows = matrix.nb_subrows;
	gpu_buffer = matrix.gpu_buffer;
}

void Matrix::load_from_cpu(DataBlock &input)
{
	cl_int error;

	if (nb_subelements() != input.size())
		throw std::runtime_error("Matrix - Incompatible size when loading from CPU into GPU");
	error = clEnqueueWriteBuffer(gpu.command_queue, gpu_buffer, CL_TRUE, start_offset() * sizeof(cl_float), nb_subelements() * sizeof(cl_float), input.cpu_buffer, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error writing into GPU memory");
}

DataBlock Matrix::fetch() const
{
	cl_int error;

	DataBlock result(nb_subrows, nb_columns);
	if (result.cpu_buffer == nullptr)
		throw std::runtime_error("Matrix - whoooot ??");

	error = clEnqueueReadBuffer(gpu.command_queue, gpu_buffer, CL_TRUE, start_offset() * sizeof(cl_float), nb_subelements() * sizeof(cl_float), result.cpu_buffer, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error reading from GPU memory");

	return result;
}

cl_float Matrix::norm() const // Euclidian norm of the matrix
{
	cl_int error = clblasSuccess;
	Matrix scratch(1, (cl_uint)(2 * nb_elements()));
	Matrix result(1, 1);
	DataBlock res_img;
	cl_float ret;

	error = clblasSnrm2(nb_elements(), result.gpu_buffer, 0, gpu_buffer, 0, 1, scratch.gpu_buffer, 1, &gpu.command_queue, 0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error calculating euclidian norm");

	res_img = result.fetch();
	ret = res_img.cpu_buffer[0];
	return ret;
}

cl_float Matrix::absmax() const // Max absolute value of all elements in Matrix
{
	cl_int error;
	Matrix scratch(1, (cl_uint)(2 * nb_elements()));
	cl_mem index_buffer;
	cl_uint index;
	cl_float result;

	index_buffer = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &error);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error allocating GPU memory");

	error = clblasiSamax(nb_elements(), index_buffer, 0, gpu_buffer, 0, 1, scratch.gpu_buffer, 1, &gpu.command_queue, 0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error calculating max absolute index");

	error = clEnqueueReadBuffer(gpu.command_queue, index_buffer, CL_TRUE, 0, sizeof(cl_uint), &index, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error reading from GPU memory");

	index--;
	clReleaseMemObject(index_buffer);

	error = clEnqueueReadBuffer(gpu.command_queue, gpu_buffer, CL_TRUE, index * sizeof(cl_float), sizeof(cl_float), &result, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error reading from GPU memory");

	return abs(result);
}

void Matrix::create()
{
	cl_int error;
	// cout << "Allocating " << nb_elements() * sizeof(cl_float) << " for " << this << endl;
	if (nb_elements() != 0)
	{
		gpu_buffer = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, nb_elements() * sizeof(cl_float), nullptr, &error);
		if (error != CL_SUCCESS)
			throw std::runtime_error("Matrix - error allocating GPU memory");
	}
	// total_mem_alloc += nb_elements() * sizeof(cl_float);
	// cout << "Memory used : " << total_mem_alloc << endl;
}

void Matrix::release()
{
	// cout << "Releasing " << nb_elements() * sizeof(cl_float) << " for " << this << endl;
	if (gpu_buffer)
		clReleaseMemObject(gpu_buffer);
	// total_mem_alloc -= nb_elements() * sizeof(cl_float);
	// cout << "Memory used : " << total_mem_alloc << endl;
}

void Matrix::is_deep_copy_of(const Matrix &matrix) // deep copy
{
	cl_int error;

	if (nb_elements() != matrix.nb_subelements())
		throw std::runtime_error("Matrix - incompatible matrix sizes for deep copy");
	error = clEnqueueCopyBuffer(gpu.command_queue, matrix.gpu_buffer, gpu_buffer, matrix.start_offset() * sizeof(cl_float), 0, matrix.nb_subelements() * sizeof(cl_float), 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - error copying in GPU memory");
}

void Matrix::select_subset(cl_uint start, cl_uint nbr)
{
	start_subrow = start;
	nb_subrows = nbr;
}

void Matrix::select_full_set()
{
	start_subrow = 0;
	nb_subrows = nb_rows;
}

void Matrix::is_transp_of(const Matrix &matrix)
{
	cl_int error;
	cl_uint sub_offset;

	if (nb_elements() != matrix.nb_subelements())
		throw std::runtime_error("Matrix - incompatible sizes for matrix transposition");

	sub_offset = (cl_uint)matrix.start_offset();
	error = clSetKernelArg(gpu.transpose_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error = clSetKernelArg(gpu.transpose_kernel, 1, sizeof(cl_mem), &matrix.gpu_buffer);
	error = clSetKernelArg(gpu.transpose_kernel, 2, sizeof(cl_uint), &sub_offset);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting transposition kernel arguments ");

	// Launching kernel
	size_t global_ws[2] = {matrix.nb_subrows, matrix.nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.transpose_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	// error |= clFinish(gpu.command_queue);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching transposition kernel ");
}

void Matrix::scale_by(cl_float scalar)
{
	cl_int error;

	error = clblasSscal(nb_subelements(), scalar, gpu_buffer, start_offset(), 1,
											1, &gpu.command_queue,
											0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error performing scalar product");
}

void Matrix::add_scaled(const Matrix &input, cl_float scalar)
{
	cl_int error;

	if (nb_subelements() != input.nb_subelements())
		throw std::runtime_error("Matrix - incompatible matrix sizes for addition");

	error = clblasSaxpy(input.nb_subelements(), scalar, input.gpu_buffer, input.start_offset(), 1,
											gpu_buffer, start_offset(), 1,
											1, &gpu.command_queue,
											0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error performing addition");
}

void Matrix::is_mult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_multiplied_scaled(0, first, second, 1);
}

void Matrix::add_mult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_multiplied_scaled(1, first, second, 1);
}

void Matrix::sub_mult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_multiplied_scaled(1, first, second, -1);
}

void Matrix::is_transmult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_transmult_scaled(0, first, second, 1);
}

void Matrix::is_multrans_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_multrans_scaled(0, first, second, 1);
}

void Matrix::add_transmult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_transmult_scaled(1, first, second, 1);
}

void Matrix::sub_transmult_of(const Matrix &first, const Matrix &second)
{
	scale_then_add_transmult_scaled(1, first, second, -1);
}

void Matrix::scale_then_add_multiplied_scaled(cl_float self_scalar,
																							const Matrix &X, const Matrix &Y, cl_float mult_scalar)
{
	cl_int error;

	if (X.nb_columns != Y.nb_subrows)
		throw std::runtime_error("Matrix - incompatible XY matrix sizes for multiplication");
	if ((X.nb_subrows != nb_subrows) || (Y.nb_columns != nb_columns))
		throw std::runtime_error("Matrix - incompatible Z matrix size for multiplication");

	error = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, X.nb_subrows, Y.nb_columns, X.nb_columns,
											mult_scalar, X.gpu_buffer, X.start_offset(), X.nb_columns,
											Y.gpu_buffer, Y.start_offset(), Y.nb_columns,
											self_scalar, gpu_buffer, start_offset(), nb_columns,
											1, &gpu.command_queue, 0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error performing multiplication");
}

void Matrix::scale_then_add_transmult_scaled(cl_float self_scalar,
																						 const Matrix &first, const Matrix &second, cl_float mult_scalar)
{
	cl_int error;

	if (first.nb_subrows != second.nb_subrows)
		throw std::runtime_error("Matrix - incompatible XY matrix sizes for multiplication");
	if ((first.nb_columns != nb_subrows) || (second.nb_columns != nb_columns))
		throw std::runtime_error("Matrix - incompatible Z matrix size for multiplication");

	error = clblasSgemm(clblasRowMajor, clblasTrans, clblasNoTrans, first.nb_columns, second.nb_columns, first.nb_subrows,
											mult_scalar, first.gpu_buffer, first.start_offset(), first.nb_columns,
											second.gpu_buffer, second.start_offset(), second.nb_columns,
											self_scalar, gpu_buffer, start_offset(), nb_columns,
											1, &gpu.command_queue, 0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error performing multiplication");
}

void Matrix::scale_then_add_multrans_scaled(cl_float self_scalar,
																						const Matrix &first, const Matrix &second, cl_float mult_scalar)
{
	cl_int error;

	if (first.nb_columns != second.nb_columns)
		throw std::runtime_error("Matrix - incompatible XY matrix sizes for multiplication");
	if ((first.nb_subrows != nb_subrows) || (second.nb_subrows != nb_columns))
		throw std::runtime_error("Matrix - incompatible Z matrix size for multiplication");

	error = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasTrans, first.nb_subrows, second.nb_subrows, first.nb_columns,
											mult_scalar, first.gpu_buffer, first.start_offset(), first.nb_columns,
											second.gpu_buffer, second.start_offset(), second.nb_columns,
											self_scalar, gpu_buffer, start_offset(), nb_columns,
											1, &gpu.command_queue, 0, nullptr, nullptr);
	if (error != clblasSuccess)
		throw std::runtime_error("Matrix - error performing multiplication");
}

void Matrix::fill_with_float(cl_float value)
{
	cl_int error;

	error = clSetKernelArg(gpu.fill_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.fill_kernel, 1, sizeof(cl_float), &value);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting fill kernel arguments ");

	// Launching kernel
	size_t global_ws[2] = {nb_rows, nb_columns}; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.fill_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);

	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching fill kernel ");
}

void Matrix::fill_right_block(const Matrix &matrix)
{
	size_t sub_offset;
	cl_int error;

	sub_offset = matrix.start_offset();
	error = clSetKernelArg(gpu.fill_right_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.fill_right_kernel, 1, sizeof(cl_mem), &(matrix.gpu_buffer));
	error |= clSetKernelArg(gpu.fill_right_kernel, 2, sizeof(size_t), &sub_offset);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting copy insert kernel arguments ");

	size_t global_ws[2] = {matrix.nb_subrows, matrix.nb_columns}; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.fill_right_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching copy insert kernel ");
}

void Matrix::fill_left_column(cl_float value)
{
	cl_int error;

	error = clSetKernelArg(gpu.fill_left_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.fill_left_kernel, 1, sizeof(cl_uint), &nb_columns);
	error |= clSetKernelArg(gpu.fill_left_kernel, 2, sizeof(cl_float), &value);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting fill left kernel arguments ");

	size_t global_ws = nb_rows; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.fill_left_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching fill left kernel ");
}

void Matrix::fill_with_random_uniform(cl_float lo, cl_float hi)
{
	DataBlock random_nbrs(nb_rows, nb_columns);
	random_nbrs.fill_with_random_uniform(lo, hi);
	load_from_cpu(random_nbrs);
}

void Matrix::fill_with_random_gaussian(cl_float mean, cl_float stdev)
{
	DataBlock random_nbrs(nb_rows, nb_columns);
	random_nbrs.fill_with_random_gaussian(mean, stdev);
	load_from_cpu(random_nbrs);
}

void Matrix::fill_with_random_binary(cl_float prob)
{
	DataBlock random_nbrs(nb_rows, nb_columns);
	random_nbrs.fill_with_random_binary(prob);
	load_from_cpu(random_nbrs);
}

void Matrix::fill_top_row(cl_float value)
{
	cl_int error;

	error = clSetKernelArg(gpu.fill_top_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.fill_top_kernel, 1, sizeof(cl_float), &value);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Error setting fill top kernel arguments ");
	size_t global_ws = nb_columns; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.fill_top_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Error launching fill top kernel ");
}

void Matrix::left_is_avg_pxls(const Matrix &images)
{
	cl_int error;

	error = clSetKernelArg(gpu.avg_pix_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error = clSetKernelArg(gpu.avg_pix_kernel, 1, sizeof(cl_uint), &nb_columns);
	error |= clSetKernelArg(gpu.avg_pix_kernel, 2, sizeof(cl_mem), &images.gpu_buffer);
	error |= clSetKernelArg(gpu.avg_pix_kernel, 3, sizeof(cl_uint), &images.nb_rows);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Error setting avg pix kernel arguments ");
	size_t global_ws = images.nb_columns; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.avg_pix_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Error launching fill left kernel ");
}

void Matrix::apply_sampling(const Matrix &random_samples)
{
	cl_int error;
	size_t sub_offset;

	if (nb_elements() != random_samples.nb_subelements())
		throw std::runtime_error("Matrix - incompatible sizes for sampling");

	sub_offset = random_samples.start_offset();
	error = clSetKernelArg(gpu.binary_distrib_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.binary_distrib_kernel, 1, sizeof(cl_mem), &random_samples.gpu_buffer);
	error |= clSetKernelArg(gpu.binary_distrib_kernel, 2, sizeof(size_t), &sub_offset);

	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting binary distribution kernel arguments ");

	size_t global_ws[2] = {random_samples.nb_subrows, random_samples.nb_columns}; // Total number of work-items

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.binary_distrib_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching binary sampling kernel ");
}

void Matrix::apply_sigmoid()
{
	cl_int error;

	error = clSetKernelArg(gpu.sigmoid_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting sigmoid kernel arguments ");

	// Launching kernel
	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.sigmoid_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching sigmoid kernel ");
}

void Matrix::apply_relu()
{
	cl_int error;

	error = clSetKernelArg(gpu.relu_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting relu kernel arguments ");

	// Launching kernel
	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.relu_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching relu kernel ");
}

void Matrix::apply_softmax()
{
	cl_int error;

	error = clSetKernelArg(gpu.softmax_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.softmax_kernel, 1, sizeof(cl_uint), &nb_columns);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting softmax kernel arguments ");

	// Launching kernel
	size_t global_ws = nb_rows;

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.softmax_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching softmax kernel ");
}

void Matrix::apply_square()
{
	cl_int error;

	error = clSetKernelArg(gpu.square_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting square kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.square_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching square kernel ");
}

void Matrix::apply_sigmoid_deriv()
{
	cl_int error;

	error = clSetKernelArg(gpu.sigmoid_deriv_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting sigmoid derivation kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.sigmoid_deriv_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching sigmoid derivation kernel ");
}

void Matrix::apply_relu_deriv()
{
	cl_int error;

	error = clSetKernelArg(gpu.relu_deriv_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting sigmoid derivation kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.relu_deriv_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching sigmoid derivation kernel ");
}

void Matrix::apply_memberprod(const Matrix &other)
{
	cl_int error;
	if (*this != other)
		throw std::runtime_error("Matrix - incompatible sizes for memberwise product");

	error = clSetKernelArg(gpu.member_prod_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.member_prod_kernel, 1, sizeof(cl_mem), &other.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting memberwise product kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.member_prod_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching memberwise product kernel ");
}

void Matrix::apply_memberadd(cl_float scalar)
{
	cl_int error;

	error = clSetKernelArg(gpu.member_add_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.member_add_kernel, 1, sizeof(cl_float), &scalar);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting memberwise addition kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.member_add_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching memberwise addition kernel ");
}

void Matrix::is_sqtrmult_of(const Matrix &left, const Matrix &top)
{
	cl_int error;
	size_t sub_offset;

	if ((nb_rows != left.nb_columns) || (nb_columns != top.nb_columns) || (left.nb_subrows != top.nb_rows))
		throw std::runtime_error("Matrix - incompatible sizes for sqtr product");

	sub_offset = left.start_offset();
	error = clSetKernelArg(gpu.sqtr_prod_kernel, 0, sizeof(cl_mem), &left.gpu_buffer);
	error |= clSetKernelArg(gpu.sqtr_prod_kernel, 1, sizeof(size_t), &sub_offset);
	error |= clSetKernelArg(gpu.sqtr_prod_kernel, 2, sizeof(cl_mem), &top.gpu_buffer);
	error |= clSetKernelArg(gpu.sqtr_prod_kernel, 3, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.sqtr_prod_kernel, 4, sizeof(cl_uint), &top.nb_rows);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting sqtr product kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.sqtr_prod_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching sqtr product kernel ");
}

void Matrix::update_rolling_average(const Matrix &new_data, const Matrix &timings)
{
	cl_int error;
	if ((new_data != timings) || (*this != new_data))
		throw std::runtime_error("Matrix - incompatible matrix sizes for update of rolling averages");

	error = clSetKernelArg(gpu.rolling_avg_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.rolling_avg_kernel, 1, sizeof(cl_mem), &new_data.gpu_buffer);
	error |= clSetKernelArg(gpu.rolling_avg_kernel, 2, sizeof(cl_mem), &timings.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting rolling average kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.rolling_avg_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching rolling average kernel ");
}

void Matrix::update_learning_rates(const Matrix &mean_gradient, const Matrix &est_hessian, const Matrix &var_gradient)
{
	cl_int error;
	if ((*this != mean_gradient) || (mean_gradient != est_hessian) || (est_hessian != var_gradient))
		throw std::runtime_error("Matrix - incompatible matrix sizes for update of learning rates");

	error = clSetKernelArg(gpu.update_rates_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.update_rates_kernel, 1, sizeof(cl_mem), &mean_gradient.gpu_buffer);
	error |= clSetKernelArg(gpu.update_rates_kernel, 2, sizeof(cl_mem), &est_hessian.gpu_buffer);
	error |= clSetKernelArg(gpu.update_rates_kernel, 3, sizeof(cl_mem), &var_gradient.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting learning rates kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.update_rates_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching learning rates kernel ");
}

void Matrix::update_timings(const Matrix &mean_gradient, const Matrix &var_gradient)
{
	cl_int error;
	if ((*this != mean_gradient) || (mean_gradient != var_gradient))
		throw std::runtime_error("Matrix - incompatible matrix sizes for update of rolling averages");

	error = clSetKernelArg(gpu.update_timings_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.update_timings_kernel, 1, sizeof(cl_mem), &mean_gradient.gpu_buffer);
	error |= clSetKernelArg(gpu.update_timings_kernel, 2, sizeof(cl_mem), &var_gradient.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting timings kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.update_timings_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching timings kernel ");
}

void Matrix::apply_mean_square_update(cl_float decay_rate, const Matrix &derivatives)
{
	cl_int error;
	if (*this != derivatives)
		throw std::runtime_error("Matrix - incompatible sizes for Mean-Square update");

	error = clSetKernelArg(gpu.mean_square_update_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.mean_square_update_kernel, 1, sizeof(cl_float), &decay_rate);
	error |= clSetKernelArg(gpu.mean_square_update_kernel, 2, sizeof(cl_mem), &derivatives.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting mean square update kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.mean_square_update_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching mean square update kernel ");
}

// old function that adds derivatives scaled by mean squares, to an external Matrix - with one global learning rate
void Matrix::add_with_rms_scaling(const Matrix &derivatives, cl_float epsilon, const Matrix &mean_squares, cl_float smoothing_factor)
{
	cl_int error;
	if ((*this != derivatives) || (*this != mean_squares))
		throw std::runtime_error("Matrix - incompatible sizes for RMS add-scaling");

	error = clSetKernelArg(gpu.add_rms_scale_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.add_rms_scale_kernel, 1, sizeof(cl_mem), &derivatives.gpu_buffer);
	error |= clSetKernelArg(gpu.add_rms_scale_kernel, 2, sizeof(cl_float), &epsilon);
	error |= clSetKernelArg(gpu.add_rms_scale_kernel, 3, sizeof(cl_mem), &mean_squares.gpu_buffer);
	error |= clSetKernelArg(gpu.add_rms_scale_kernel, 4, sizeof(cl_float), &smoothing_factor);

	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting RMS add-scaling kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.add_rms_scale_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching RMS add-scaling kernel ");
}

// function that modifies the derivatives directly by scaling them with the mean squares, with individual learning rates
void Matrix::apply_rms_scaling(const Matrix &learning_rates, const Matrix &mean_squares, cl_float damping_factor)
{
	cl_int error;
	if ((*this != mean_squares) || (*this != learning_rates))
		throw std::runtime_error("Matrix - incompatible sizes for RMS scaling");

	error = clSetKernelArg(gpu.rms_scale_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.rms_scale_kernel, 1, sizeof(cl_mem), &learning_rates.gpu_buffer);
	error |= clSetKernelArg(gpu.rms_scale_kernel, 2, sizeof(cl_mem), &mean_squares.gpu_buffer);
	error |= clSetKernelArg(gpu.rms_scale_kernel, 3, sizeof(cl_float), &damping_factor);

	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting RMS scaling kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.rms_scale_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching RMS scaling kernel ");
}

void Matrix::adapt_and_clip(const Matrix &gradients, const Matrix &velocities, cl_float adapt_rate, cl_float min_rate, cl_float max_rate)
{
	cl_int error;
	if ((*this != gradients) || (*this != velocities))
		throw std::runtime_error("Matrix - incompatible sizes for rate clipping");

	error = clSetKernelArg(gpu.adapt_clip_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.adapt_clip_kernel, 1, sizeof(cl_mem), &gradients.gpu_buffer);
	error |= clSetKernelArg(gpu.adapt_clip_kernel, 2, sizeof(cl_mem), &velocities.gpu_buffer);
	error |= clSetKernelArg(gpu.adapt_clip_kernel, 3, sizeof(cl_float), &adapt_rate);
	error |= clSetKernelArg(gpu.adapt_clip_kernel, 4, sizeof(cl_float), &min_rate);
	error |= clSetKernelArg(gpu.adapt_clip_kernel, 5, sizeof(cl_float), &max_rate);

	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting rate clipping kernel arguments ");

	size_t global_ws[2] = {nb_rows, nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.adapt_clip_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching rate clipping kernel ");
}

cl_float Matrix::free_nrg(const Matrix &neurons) const // Matrix object must be a visible vector
{
	Matrix interm(nb_subrows, neurons.nb_columns);
	DataBlock result;
	cl_float ret;
	cl_int error;

	interm.is_mult_of(*this, neurons);

	error = clSetKernelArg(gpu.free_nrg_1_kernel, 0, sizeof(cl_mem), &interm.gpu_buffer);
	error |= clSetKernelArg(gpu.free_nrg_1_kernel, 1, sizeof(cl_mem), &neurons.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting free_nrg_1 kernel arguments ");

	size_t global_ws[2] = {interm.nb_rows, interm.nb_columns};

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.free_nrg_1_kernel, 2, nullptr, global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching free_nrg_1 kernel ");

	error = clSetKernelArg(gpu.free_nrg_2_kernel, 0, sizeof(cl_mem), &interm.gpu_buffer);
	error |= clSetKernelArg(gpu.free_nrg_2_kernel, 1, sizeof(cl_uint), &interm.nb_columns);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting free_nrg_2 kernel arguments ");

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.free_nrg_2_kernel, 1, nullptr, &global_ws[0], nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching free_nrg_2 kernel ");

	result = interm.fetch();
	ret = 0;
	for (cl_uint row = 0; row < interm.nb_rows; row++)
	{
		ret += result.cpu_buffer[row * interm.nb_columns];
	}
	ret /= interm.nb_rows;
	// interm.release();
	return ret;
}

cl_float Matrix::cross_entropy(const Matrix &goals, const Matrix &values)
{
	cl_float cost = 0;
	cl_int error;
	size_t sub_offset;
	Matrix cross_results(goals.nb_subrows, 1);
	DataBlock result_image;

	if (goals != values)
		throw std::runtime_error("Matrix - incompatible matrix sizes for cross-entropy computation");

	sub_offset = goals.start_offset();
	error = clSetKernelArg(gpu.cross_entropy_kernel, 0, sizeof(cl_mem), &goals.gpu_buffer);
	error |= clSetKernelArg(gpu.cross_entropy_kernel, 1, sizeof(cl_mem), &values.gpu_buffer);
	error |= clSetKernelArg(gpu.cross_entropy_kernel, 2, sizeof(cl_uint), &goals.nb_columns);
	error |= clSetKernelArg(gpu.cross_entropy_kernel, 3, sizeof(size_t), &sub_offset);
	error |= clSetKernelArg(gpu.cross_entropy_kernel, 4, sizeof(cl_mem), &cross_results.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting cross-entropy kernel arguments ");

	size_t global_ws = goals.nb_subrows;

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.cross_entropy_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching cross-entropy kernel ");
	result_image = cross_results.fetch();
	for (unsigned int i = 0; i < result_image.nb_rows; i++)
	{
		cost += result_image.cpu_buffer[i];
	}
	cost /= result_image.nb_rows;
	if (!(cost >= 0))
	{
		cost = -1;
	}
	// cross_results.release();
	return cost;
}

cl_float Matrix::error_rate(const Matrix &goals, const Matrix &outputs)
{
	cl_float result = 0;
	cl_int error;
	size_t sub_offset;
	Matrix cross_results(goals.nb_subrows, 1);
	DataBlock result_image;

	if (goals != outputs)
		throw std::runtime_error("Matrix - incompatible matrix sizes for error rate computation");

	sub_offset = goals.start_offset();
	error = clSetKernelArg(gpu.error_rate_kernel, 0, sizeof(cl_mem), &goals.gpu_buffer);
	error |= clSetKernelArg(gpu.error_rate_kernel, 1, sizeof(cl_mem), &outputs.gpu_buffer);
	error |= clSetKernelArg(gpu.error_rate_kernel, 2, sizeof(cl_uint), &goals.nb_columns);
	error |= clSetKernelArg(gpu.error_rate_kernel, 3, sizeof(size_t), &sub_offset);
	error |= clSetKernelArg(gpu.error_rate_kernel, 4, sizeof(cl_mem), &cross_results.gpu_buffer);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting error rate kernel arguments ");

	size_t global_ws = goals.nb_subrows;

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.error_rate_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching error rate kernel ");

	result_image = cross_results.fetch();
	for (unsigned int i = 0; i < result_image.nb_rows; i++)
	{
		result += result_image.cpu_buffer[i];
	}
	result /= result_image.nb_rows;
	/*
		if (result > 0.90f) {
			goals.display("goals");
			outputs.display("outputs");
			cross_results.display("!!! cross-results for error");
			result = -1;
		}
	 */
	// cross_results.release();
	return result;
}

void Matrix::shuffle_rows()
{
	cl_int error;

	/* Durstenfeld shuffling algorithm
	To shuffle an array a of n elements (indices 0..n-1):
	for i from 0 to n − 2 do
		j ← random integer such that i ≤ j < n
		exchange a[j] and a[i]
	 */
	DataBlock new_positions;
	Matrix new_rows;

	cl_uint new_nbitems = nb_rows - 1;
	new_positions = DataBlock(1, new_nbitems);
	default_random_engine generator((unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count());
	for (cl_uint i = 0; i < new_nbitems; i++)
	{
		uniform_int_distribution<int> distribution(i, new_nbitems);
		new_positions.cpu_buffer[i] = distribution(generator);
	}
	new_rows = Matrix(new_positions);

	error = clSetKernelArg(gpu.shuffle_rows_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.shuffle_rows_kernel, 1, sizeof(cl_mem), &new_rows.gpu_buffer);
	error |= clSetKernelArg(gpu.shuffle_rows_kernel, 2, sizeof(cl_uint), &new_nbitems);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting shuffle rows kernel arguments ");

	// Launching kernel
	size_t global_ws = nb_columns;

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.shuffle_rows_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching shuffle rows kernel ");
}

void Matrix::standardize()
{
	cl_int error;

	error = clSetKernelArg(gpu.standardize_kernel, 0, sizeof(cl_mem), &gpu_buffer);
	error |= clSetKernelArg(gpu.standardize_kernel, 1, sizeof(cl_uint), &nb_rows);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error setting standardize kernel arguments ");

	// Launching kernel
	size_t global_ws = nb_columns;

	error = clEnqueueNDRangeKernel(gpu.command_queue, gpu.standardize_kernel, 1, nullptr, &global_ws, nullptr, 0, nullptr, nullptr);
	if (error != CL_SUCCESS)
		throw std::runtime_error("Matrix - Error launching standardize kernel ");
}

void Matrix::display(std::string name) const
{
	DataBlock content;
	cl_uint position;

	std::cout << name << " :" << std::endl;
	content = fetch();
	position = 0;

	for (cl_uint row = 0; row < content.nb_rows; row++)
	{
		for (cl_uint col = 0; col < content.nb_cols; col++)
		{
			std::cout << content.cpu_buffer[position] << " ; ";
			position++;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
