//
//  Kernels.cl
//  deep learning
//
//  Created by Sébastien Raffray on 12/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

__kernel void sigmoid (__global float* matrix)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);
    const uint cur_col = get_global_id(1);
    const size_t position = cur_row * nb_cols + cur_col;
	matrix[position] = native_recip(1 + native_exp(-matrix[position]));
}

__kernel void relu (__global float* matrix)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	if (matrix[position] <= 0.0f)
		matrix[position] = 0.0f;
}

__kernel void softmax (__global float* matrix, // applique softmax; met les sommes(exp(xj)) dans la 1ere colonne
					   const uint nb_cols)		// algo qui évite division par zéro qd les logits sont très négatifs
{
	const uint cur_row = get_global_id(0);      // 1D kernel : rangées
	size_t sum_pos, end_pos;
	float max_val;
	
	sum_pos = cur_row * nb_cols;
	end_pos = sum_pos + nb_cols;

	max_val = matrix[sum_pos + 1];
	for (uint cur_pos = sum_pos + 2; cur_pos < end_pos; cur_pos++) {
		if (matrix[cur_pos] > max_val)
			max_val = matrix[cur_pos];
	}
	
	matrix[sum_pos] = 0;
	for (uint cur_pos = sum_pos + 1; cur_pos < end_pos; cur_pos++) {
		matrix[cur_pos] -= max_val;
		matrix[cur_pos] = native_exp(matrix[cur_pos]);
		matrix[sum_pos] += matrix[cur_pos];
	}
	
	for (uint cur_pos = sum_pos + 1; cur_pos < end_pos; cur_pos++) {
		matrix[cur_pos] = native_divide(matrix[cur_pos], matrix[sum_pos]);
	}
}

__kernel void square(__global float* matrix)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
    const uint cur_col = get_global_id(1);
    size_t position = cur_row * nb_cols + cur_col;
    matrix[position] *= matrix[position];
}


__kernel void exponential (__global float* matrix)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
    const uint cur_col = get_global_id(1);
    size_t position = cur_row * nb_cols + cur_col;
    matrix[position] = native_exp(matrix[position]);
}

__kernel void sum_reduce (__global float* matrix,
                          const uint nb_cols)
{
    const uint cur_row = get_global_id(0);      // 1D kernel : rangées
    size_t sum_pos, end_pos;
    sum_pos = cur_row * nb_cols;
    end_pos = sum_pos + nb_cols;
    for (uint cur_pos = sum_pos + 1; cur_pos < end_pos; cur_pos++) {
        matrix[sum_pos] += matrix[cur_pos];
    }
}

__kernel void divide (__global float* matrix)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
    const uint cur_col = get_global_id(1);
    size_t div_pos, cur_pos;
    if (cur_col > 0) {
        div_pos = cur_row * nb_cols;
        cur_pos = div_pos + cur_col;
        matrix[cur_pos] = native_divide(matrix[cur_pos], matrix[div_pos]);
    }
}

__kernel void sigmoid_deriv (__global float* y_values)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
    const size_t position = cur_row * nb_cols + cur_col;
	y_values[position] *= (1 - y_values[position]);
}

__kernel void relu_deriv  (__global float* y_values)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	if (y_values[position] > 0.0f)
		y_values[position] = 1.0f;
}

__kernel void member_prod (__global float* dest,
						   __global const float* source)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	dest[position] *= source[position];
}

__kernel void member_add (__global float* dest,
						  const float scalar)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	dest[position] += scalar;
}

__kernel void sqtr_prod (__global const float* left_matrix, // dest = (leftT)^2 * top, avec carré au sens du produit membre à membre
						const size_t left_offset,
						 __global const float* top_matrix,	// fonctionne en mini_batch également, avec common_dim vecteurs additionnés
						 __global float* dest_matrix,
						 const uint common_dim)
{
 	const uint nb_rows = get_global_size(0);
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t dest_position = cur_row * nb_cols + cur_col;
	size_t left_position, top_position;
	dest_matrix[dest_position] = 0;
	for (uint dim = 0; dim < common_dim; dim++) {
		left_position = left_offset + dim * nb_rows + cur_row;
		top_position = dim * nb_cols + cur_col;
		dest_matrix[dest_position] += pown(left_matrix[left_position], 2) * top_matrix[top_position];
	}
}

__kernel void rolling_avg (__global float* result,
						   __global const float* new_data,
						   __global const float* timings)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	const float gamma = native_recip(timings[position]);
	result[position] *= (1 - gamma);
	result[position] += gamma * new_data[position];
}

__kernel void update_rates (__global float* learning_rate,
						   __global const float* mean_gradient,
						   __global const float* est_hessian,
						   __global const float* var_gradient)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	learning_rate[position] = native_divide(pown(mean_gradient[position], 2) , 0.01f + est_hessian[position]*var_gradient[position]);
}

__kernel void update_timings (__global float* timings,
							  __global const float* mean_gradient,
							  __global const float* var_gradient)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	timings[position] *= 1 - native_divide(pown(mean_gradient[position], 2), 0.01f + var_gradient[position]);
	timings[position] += 1;
}

__kernel void mean_square_update (__global float* mean_squares,
								  const float decay_rate,
								  __global const float* derivatives)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	mean_squares[position] = decay_rate * mean_squares[position] + (1-decay_rate) * derivatives[position] * derivatives[position];
}

// old kernel that adds rms_scaling of derivatives to neurons (does not touch derivatives) with one global learning rate
__kernel void add_rms_scale (__global float* neurons,
							 __global const float* derivatives,
							 const float epsilon,
							 __global const float* mean_squares,
							 const float damping_factor)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	neurons[position] += native_divide(epsilon * derivatives[position], native_sqrt(mean_squares[position]) + damping_factor);
}


// kernel that scales the derivatives matrix itself (touches derivatives) with individual learning rates
__kernel void rms_scale (__global float* derivatives,
						 __global const float* learning_rates,
						 __global const float* mean_squares,
						 const float damping_factor)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	derivatives[position] *= native_divide(learning_rates[position], native_sqrt(mean_squares[position]) + damping_factor);
}

__kernel void adapt_clip (__global float* learning_rates,
						  __global const float* gradients,
						  __global const float* velocities,
						  const float adapt_rate,
						  const float min_rate,
						  const float max_rate)
{
	const uint nb_cols = get_global_size(1);
	const uint cur_row = get_global_id(0);      // 2D kernel : rangées, colonnes
	const uint cur_col = get_global_id(1);
	const size_t position = cur_row * nb_cols + cur_col;
	if (gradients[position] * velocities[position] >= 0)
	{
		learning_rates[position] = fmin(learning_rates[position] * (1 + adapt_rate), max_rate);
	}
	else
	{
		learning_rates[position] = fmax(learning_rates[position] * (1 - adapt_rate), min_rate);
	}
}

__kernel void cross_entropy (__global const float* goals,
							 __global const float* values,
							 const uint nb_cols,
							 const size_t goals_offset,
							 __global float* cross_results) // cross results est un vecteur vertical
{
	const uint cur_row = get_global_id(0);		// 1D kernel : rangées
	size_t position = cur_row * nb_cols;
	float result = 0;
	for (uint cur_col = 1; cur_col < nb_cols; cur_col++) {
		position++;
		if (goals[goals_offset + position] > 0) { // skippe les valeurs nulles : mult par 0 inutile
			if (values[position] > 1E-05f) { // évite le calcul de log(0)
				result += goals[goals_offset + position] * native_log(values[position]);
			}
			else {
				result += goals[goals_offset + position] * -100; // auquel cas assume 1*log(~0) = -10;
			}
		}
	}
	cross_results[cur_row] = -result;
}

__kernel void error_rate (__global const float* goals,
						  __global const float* outputs,
						  const uint nb_cols,
						  const size_t goals_offset,
						  __global float* cross_results) // cross results est un vecteur vertical
{
	const uint cur_row = get_global_id(0);		// 1D kernel : rangées
	size_t position = cur_row * nb_cols;
	float max_goal = 0;
	uint maxarg_goals = 0;
	float max_output = 0;
	uint maxarg_outputs = 10;
	for (uint cur_col = 0; cur_col < nb_cols - 1; cur_col++) {
		position++;
		if (goals[goals_offset + position] > max_goal) {
			max_goal = goals[goals_offset + position];
			maxarg_goals = cur_col;
		}
		if (outputs[position] > max_output) {
			max_output = outputs[position];
			maxarg_outputs = cur_col;
		}
	}
	if (maxarg_goals == maxarg_outputs)
		cross_results[cur_row] = 0;
	else cross_results[cur_row] = 1;
}

__kernel void fill (__global float* matrix,
                    const float value)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);
    const uint cur_col = get_global_id(1);
    const size_t position = cur_row * nb_cols + cur_col;
    matrix[position] = value;
}

__kernel void fill_left (__global float* matrix,
                         const uint nb_cols,
                         const float value)
{
    const uint cur_row = get_global_id(0);
    const size_t position = cur_row * nb_cols;
    matrix[position] = value;
}

__kernel void fill_top (__global float* matrix,
                         const float value)
{
    const uint cur_col = get_global_id(0); // Paramètre 1D = nb colonnes de matrix
    matrix[cur_col] = value;
}

__kernel void fill_right (__global float* dest,
						  __global const float* source,
						  const size_t src_offset)
{
	const uint nb_cols = get_global_size(1);                // Le nb total de colonnes de la matrice source
	const uint cur_row = get_global_id(0);                  // La rangée dans le bloc de rangées à copier
	const uint cur_col = get_global_id(1);                  // La colonne actuelle dans la matrice source
	const size_t src_pos = src_offset + (cur_row  * nb_cols) + cur_col;
	const size_t dest_pos = cur_row * (nb_cols + 1) + (cur_col + 1);
	dest[dest_pos] = source[src_pos];
}


__kernel void avg_pix (__global float* dest_neurons,
                       const uint nbr_neurons,
                       __global const float* src_images,
                       const uint nbr_images)
{
    const uint src_col = get_global_id(0); // Paramètre 1D = colonnes de la matrice image [0 - nbr_pixels+1]
    const uint src_cols = get_global_size(0);  // nbr_pixels+1 = nb colonnes de la matrice images
    const uint dest_row = src_col;
    float proba = 0;
    for (uint src_row = 0; src_row < nbr_images; src_row++) {
        proba += src_images[src_row * src_cols + src_col];
    }
    proba /= nbr_images;
    if (proba == 0) {
        dest_neurons[dest_row * nbr_neurons] = -4;
    }
    else if (proba == 1) {
        dest_neurons[dest_row * nbr_neurons] = 4;
    }
    else {
        dest_neurons[dest_row * nbr_neurons] = log(proba / (1-proba));
    }
}


 __kernel void binary_distrib (__global float* dest,
                               __global const float* sample,
                               const size_t sample_offset)
 {
     const uint nb_cols = get_global_size(1);
     const uint cur_row = get_global_id(0);
     const uint cur_col = get_global_id(1);
     const size_t position = cur_row * nb_cols + cur_col;
     if (sample[position + sample_offset] < dest[position])
         dest[position] = 1.0f;
     else
         dest[position] = 0.0f;
 }

__kernel void transpose (__global float* dest,
                         __global float* src,
						 const uint src_offset)
{
    const uint nb_rows = get_global_size(0);
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);
    const uint cur_col = get_global_id(1);
    const size_t straight_position = src_offset + cur_row * nb_cols + cur_col;
    const size_t transposed_position = cur_col * nb_rows + cur_row;
    dest[transposed_position] = src[straight_position];
}

__kernel void free_nrg_1 (__global float* interm,
                          __global const float* corner_neuron)
{
    const uint nb_cols = get_global_size(1);
    const uint cur_row = get_global_id(0);
    const uint cur_col = get_global_id(1);
    const size_t position = cur_row * nb_cols + cur_col;
    if (cur_col == 0)
        interm[position] = - interm[position] + corner_neuron[0];
    else
    {
        interm[position] = - log(1 + exp(interm[position]));
    }
}

__kernel void free_nrg_2 (__global float* interm,
                          const uint nb_cols)
{
    const uint cur_row = get_global_id(0);
    const size_t position = cur_row * nb_cols;
    for (uint cur_col = 1; cur_col < nb_cols; cur_col++)
    {
        interm[position] += interm[position + cur_col];
    }
}

__kernel void shuffle_rows(__global float* matrix,
					  __global const float* new_rows,
					  const uint nb_rows)
{
	const uint nb_cols = get_global_size(0);
	const uint cur_col = get_global_id(0);
	size_t cur_pos;
	size_t new_pos;
	float saved_value;
	for (uint cur_row = 0; cur_row < nb_rows; cur_row++) {
		cur_pos = cur_row * nb_cols + cur_col;
		saved_value = matrix[cur_pos];
		new_pos = new_rows[cur_row] * nb_cols + cur_col;
		matrix[cur_pos] = matrix[new_pos];
		matrix[new_pos] = saved_value;
	}
}

__kernel void standardize(__global float* matrix,
						  const uint nb_rows)
{
	const uint nb_cols = get_global_size(0);
	const uint cur_col = get_global_id(0);
	size_t pos;
	float mean = 0;
	float stdev = 0;
	// Calcule la moyenne
	for (uint cur_row = 0; cur_row < nb_rows; cur_row++) {
		pos = cur_row * nb_cols + cur_col;
		mean += matrix[pos];
	}
	mean /= nb_rows;
	// Calcule la deviation standard
	for (uint cur_row = 0; cur_row < nb_rows; cur_row++) {
		pos = cur_row * nb_cols + cur_col;
		stdev += pown(matrix[pos] - mean, 2);
	}
	stdev /= nb_rows;
	stdev = native_sqrt(stdev);
	// Standardise la colonne
	for (uint cur_row = 0; cur_row < nb_rows; cur_row++) {
		pos = cur_row * nb_cols + cur_col;
		matrix[pos] -= mean;
		if (stdev != 0) {
			matrix[pos] /= stdev;
		}
	}
}