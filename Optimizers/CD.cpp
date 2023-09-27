//
//  CD.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 13/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "CD.h"
/*
void	CD::generative_training(Matrix &input_images, const cl_uint nbr_epochs)
{
	cl_uint nbr_imgs;
	cl_uint max_img;
	cl_float learning_rate = 0.005f;
	Matrix  random_nbrs;
	cl_uint batch_size = 20;
	Matrix  hidden_units(batch_size, nbr_outputs + 1);
	Matrix  CD(nbr_inputs + 1, nbr_outputs + 1);
	Matrix  reconstructed_units(batch_size, nbr_inputs + 1);
	chrono::time_point<chrono::high_resolution_clock> start_time, end_time;
	chrono::duration<double> elapsed_seconds;
	
	// initialise random table used for binary sampling in CD procedure
	nbr_imgs = input_images.get_rows();
	random_nbrs = Matrix(nbr_imgs, nbr_outputs + 1);
	max_img = (input_images.get_rows() / batch_size) * batch_size;
	
	for (cl_uint epoch = 0; epoch < nbr_epochs; epoch++) {
		start_time = chrono::high_resolution_clock::now();
		cout << "*** Epoch #" << epoch << "...";
		random_nbrs.select_full_set();
		random_nbrs.fill_with_random_uniform(0, 1);
		for (cl_uint cur_img = 0; cur_img < max_img; cur_img += batch_size) {
			// Contrastive Divergence Algorithm
			input_images.select_subset(cur_img, batch_size);
			
			// First bottom-up pass
			//forward_pass(mini_batch, hidden_units);
			forward_pass(input_images, hidden_units);
			CD.is_transmult_of(input_images, hidden_units);
			
			// sets hidden units to binary values before top-down pass
			random_nbrs.select_subset(cur_img, batch_size);
			hidden_units.apply_sampling(random_nbrs);
			hidden_units.fill_left_column(1);
			do_top_down(hidden_units, reconstructed_units);
			
			// calculates CD+ - CD- based on probabilities after bottom-up pass
			reconstructed_units.fill_left_column(1);
			forward_pass(reconstructed_units, hidden_units);
			CD.sub_transmult_of(reconstructed_units, hidden_units);
			
			// updates weights
			
			weights.add_scaled(CD, learning_rate / (cl_float)batch_size);
		}
		end_time = chrono::high_resolution_clock::now();
		elapsed_seconds = end_time - start_time;
		cout << " took " << elapsed_seconds.count() << " seconds." << endl;
		// calculates cost function after top-down pass
		reconstructed_units.fill_left_column(1);
		reconstructed_units -= input_images;
		cout << "Cost : " << reconstructed_units.norm() << endl << endl;
		
		
		//f1 = cur_layer.Sigmoid->training_energy();
		//f2 = cur_layer.Sigmoid->validation_energy();
		
		
		input_images.select_subset(0, 100);
		Matrix inputs = input_images;
		display_and_replace(inputs);
		
	}
	input_images.select_full_set();
}


void		CD::pre_train(const DataBlock& input_images,
							   cl_uint nbr_epochs)
{
	std::cout << "Launching training" << endl;
	Matrix	cur_input;
	Matrix	cur_output;
	Sigmoid		*cur_Sigmoid;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
	std::chrono::duration<double> elapsed_seconds;
	
	images_to_data(input_images, cur_input);
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		start_time = std::chrono::high_resolution_clock::now();
		
		cur_Sigmoid = dynamic_cast<Sigmoid*>(layers[i]);
		if (cur_Sigmoid != nullptr) {
			std::cout << endl << "*** Training " << cur_Sigmoid->get_title() << ", " << cur_Sigmoid->get_nbr_inputs() << "x" << cur_Sigmoid->get_nbr_outputs() << endl;
			cur_Sigmoid->init_weights(cur_input);
			cur_Sigmoid->generative_training(cur_input, nbr_epochs);
			end_time = std::chrono::high_resolution_clock::now();
			elapsed_seconds = end_time - start_time;
			std::cout << "*** Finished training Sigmoid in " << elapsed_seconds.count() << " seconds." << std::endl;
			
			cur_output = Matrix(cur_input.get_rows(), cur_Sigmoid->get_nbr_outputs() + 1);
			cur_Sigmoid->forward_pass(cur_input, cur_output);
			swap(cur_input, cur_output);
			cur_input.fill_left_column(1);
		}
	}
}
 */
// Mnist database: images 28x28, taille du vecteur 784
// réseau de 225 neurones
//
// Matrice Wij = matrice des poids des neurones (i: nbr lignes = 784, j: nbr colonnes = 225)
// Vecteur Bi = vecteur des biais visibles (i: nbr pixel de l'image = 784)
// Vecteur Cj = vecteur des biais cachés (j: nbr neurones = 225)
// l = learning rate (taux d'apprentissage) = 0.1
//
// initialiser Wij avec des valeurs random positives faibles (environ 0.1)
// utiliser une gaussienne centrée sur zero de déviation 0.01
// initialiser Cj à 0
// initialiser Bi: bi = log(pi / (1 - pi))
// avec pi = proportion des pixels i qui sont allumés dans l'ensemble des images
//
// pour chaque cycle (de 1 à 20):
//		Matrice CD+ = 0
//		Matrice CD- = 0
// 		pour chaque image
//			la mettre dans un vecteur
//			avec ce vecteur, effectuer contrastive divergence:
//				effectuer Gibbs sampling(k etapes, ici k = 1):
//					1. phase montante:
//						partir du vecteur visible
//						le multiplier par la matrice des poids
//						appliquer la fonction sigmoide
//						on obtient le vecteur caché
//						calculer la matrice (Si.Sj)0
//					2. phase descendante
//						partir du vecteur caché
//						transposer la matrice des poids
//						multiplier le vecteur caché par la matrice transposée
//						appliquer la fonction sigmoide
//						on obtient le vecteur visible reconstruit
//					3. phase montante:
//						partir du vecteur visible reconstruit (phase 2)
//						le multiplier par la matrice des poids
//						appliquer la fonction sigmoide
//						on obtient le vecteur caché
//						calculer la matrice (Si.Sj)1
//				CD+ = CD+ + (Si.Sj)0
//				CD- = CD- + (Si.Sj)1
//		fin cycle image
//		CD+ = CD+ / nbr images
//		CD- = CD- / nbr images
//		CD = CD+ - CD-
//		W = W + l.CD
//		afficher W: les 225 neurones (par colonne), dans un carré de 15x15
//			chacun des 784 points de la colonne représente un pixel caché
//		fin du cycle
//

// pixel put degueux en c
// olmlx loop

//
