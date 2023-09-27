//
//  main.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 08/12/2014.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Optimizers.h"
#include "Utilities/Clustering.h"


int		main(int argc, char **argv) {
	
/*
	 // CODE FOR THE MNIST DATASET
	 cl_uint	nbr_pixels_per_image;
	 Matrix	train_images;
	 Matrix	train_labels;
	 Matrix	test_images;
	 Matrix	test_labels;
	 
	 train_images = Extractor::get_images("Image database/train-images.idx3-ubyte", 60000);
	 Extractor::scale(train_images, 1.0f / 255);
	 //Extractor::shift(train_images, -0.5f);
	 train_labels = Extractor::get_labels("Image database/train-labels.idx1-ubyte", 60000);
	 test_images = Extractor::get_images("Image database/t10k-images.idx", 10000);
	 Extractor::scale(test_images, 1.0f / 255);
	 //Extractor::shift(test_images, -0.5f);
	 test_labels = Extractor::get_labels("Image database/t10k-labels.idx1-ubyte", 10000);
	 
	 nbr_pixels_per_image = train_images.get_cols() - 1;
	 
	 Network		neural_net(nbr_pixels_per_image);
	 //neural_net.add_dropout(0.8f);
	 neural_net.add_relu(1000);
	 neural_net.add_dropout();
	 neural_net.add_relu(1000);
	 neural_net.add_dropout();
	 //neural_net.add_relu(900);
	 //neural_net.add_dropout();
	 neural_net.add_softmax_finish();
	 
	 
	 RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	 
	 optimizer.initialise(200, neural_net);
	 optimizer.learn(train_images, train_labels, test_images, test_labels, 1000);
	 //
*/
	
	// CODE FOR THE METADATA
	

// SMALL TEST - 100 FIRST FILES OF METADATA BASE
	 
	std::string file_path = "Image database/outputfile.mta";
	Matrix metadata_set;
	
	metadata_set = Extractor::get_metadata(file_path);
	metadata_set.select_subset(0, 100);
	metadata_set = metadata_set;
	Extractor::standardize(metadata_set);
	
	Network autoencoder(metadata_set.get_cols() - 1);
	autoencoder.set_display();
	if (Extractor::network_exists("SmallTest")) {
		autoencoder.logs.add("Loading existing network");
		autoencoder.load("SmallTest");
	}
	else {
		autoencoder.logs.add("Creating new network");
		autoencoder.add_layer(LAYER_RELU, 100);
		autoencoder.add_layer(LAYER_MEANSQR, metadata_set.get_cols() - 1);
	}
	Adam optimizer = Adam();
	//RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	//RmsProp optimizer = RmsProp();
	
	optimizer.initialise(100, autoencoder);
	optimizer.autoencode(metadata_set, 2000);
	
	autoencoder.save("SmallTest");
	Matrix outputs;
	metadata_set = Extractor::get_metadata(file_path);
	metadata_set.select_subset(0, 100);
	metadata_set = metadata_set;
	outputs = autoencoder.get_outputs_in_layer(0, metadata_set);
	Extractor::save_matrix("Backups/SmallTest_outputs", outputs);
	/*
	 
// GREEDY TRAINING - STEP 1/3
	std::string file_path = "Image database/outputfile.mta";
	Matrix metadata_set;
	
	metadata_set = Extractor::get_metadata(file_path);
	//metadata_set.select_subset(0, 100);
	//metadata_set = metadata_set;
	Extractor::standardize(metadata_set);
	
	Network autoencoder(metadata_set.get_cols() - 1);
	autoencoder.set_display();
	if (Extractor::network_exists("Test")) {
		autoencoder.logs.add("Loading existing network");
		autoencoder.load("Test");
	}
	else {
		autoencoder.logs.add("Creating new network");
		autoencoder.add_layer(LAYER_RELU, 100);
		autoencoder.add_layer(LAYER_MEANSQR, metadata_set.get_cols() - 1);
	}
	Adam optimizer = Adam();
	//RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	//RmsProp optimizer = RmsProp();
	
	optimizer.initialise(100, autoencoder);
	optimizer.autoencode(metadata_set, 200);
	
	autoencoder.save("Test");
	Matrix outputs;
	metadata_set = Extractor::get_metadata(file_path);
	outputs = autoencoder.get_outputs_in_layer(0, metadata_set);
	Extractor::save_matrix("Backups/Test_layer0_outputs", outputs);
	 */
	
	/*
// GREEDY TRAINING - STEP 2/3
	Matrix features_vector = Extractor::load_matrix("Backups/Test_layer0_outputs");
	features_vector.fill_left_column(1);
	Network autoencoder(features_vector.get_cols() - 1);
	autoencoder.set_display();
	if (Extractor::network_exists("Test2")) {
		autoencoder.logs.add("Loading existing network");
		autoencoder.load("Test2");
	}
	else {
		autoencoder.logs.add("Creating new network");
		autoencoder.add_layer(LAYER_RELU, 20);
		autoencoder.add_layer(LAYER_MEANSQR, features_vector.get_cols() - 1);
	}
	Adam optimizer = Adam();
	//RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	//RmsProp optimizer = RmsProp();
	
	optimizer.initialise(100, autoencoder);
	optimizer.autoencode(features_vector, 2000);
	
	autoencoder.save("Test2");
	Matrix outputs;
	metadata_set = Extractor::get_metadata(file_path);
	outputs = autoencoder.get_outputs_in_layer(0, features_vector);
	Extractor::save_matrix("Backups/Test2_layer0_outputs", outputs);
	*/
	
	/*
// GREEDY TRAINING - STEP 3/3
	std::string file_path = "Image database/outputfile.mta";
	Matrix metadata_set;
	
	metadata_set = Extractor::get_metadata(file_path);
	Extractor::standardize(metadata_set);
	
	Network autoencoder(metadata_set.get_cols() - 1);
	autoencoder.set_display();
	if (Extractor::network_exists("Test3")) {
		autoencoder.logs.add("Loading existing network");
		autoencoder.load("Test3");
	}
	else {
		Matrix weights;
		autoencoder.logs.add("Creating new network");
		
		autoencoder.add_layer(LAYER_RELU, 100);
		weights = Extractor::load_matrix("Backups/Test_layer0.wgt");
		autoencoder.get_weights_in_layer(0) = weights;
		
		autoencoder.add_layer(LAYER_RELU, 20);
		weights = Extractor::load_matrix("Backups/Test2_layer0.wgt");
		autoencoder.get_weights_in_layer(1) = weights;
		
		autoencoder.add_layer(LAYER_RELU, 100);
		weights = Extractor::load_matrix("Backups/Test2_layer1.wgt");
		autoencoder.get_weights_in_layer(2) = weights;
		
		autoencoder.add_layer(LAYER_MEANSQR, metadata_set.get_cols() - 1);
		weights = Extractor::load_matrix("Backups/Test_layer1.wgt");
		autoencoder.get_weights_in_layer(3)= weights;
	}
	Adam optimizer = Adam();
	//RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	//RmsProp optimizer = RmsProp();
	
	optimizer.initialise(100, autoencoder);
	optimizer.autoencode(metadata_set, 200);
	
	autoencoder.save("Test3");
	Matrix outputs;
	metadata_set = Extractor::get_metadata(file_path);
	outputs = autoencoder.get_outputs_in_layer(1, metadata_set);
	Extractor::save_matrix("Backups/Greedy_outputs", outputs);
	 */
	
	/*

// NOT GREEDY TRAINING - ALL IN ONE STEP
	std::string file_path = "Image database/outputfile.mta";
	Matrix metadata_set;
	
	metadata_set = Extractor::get_metadata(file_path);
	Extractor::standardize(metadata_set);
	
	Network autoencoder(metadata_set.get_cols() - 1);
	autoencoder.set_display();
	if (Extractor::network_exists("2Dcompress")) {
		autoencoder.logs.add("Loading existing network");
		autoencoder.load("2Dcompress");
	}
	else {
		autoencoder.logs.add("Creating new network");
		autoencoder.add_layer(LAYER_RELU, 100);
		autoencoder.add_layer(LAYER_RELU, 20);
		autoencoder.add_layer(LAYER_RELU, 2);
		autoencoder.add_layer(LAYER_RELU, 20);
		autoencoder.add_layer(LAYER_RELU, 100);
		autoencoder.add_layer(LAYER_MEANSQR, metadata_set.get_cols() - 1);
	}
	Adam optimizer = Adam();
	RmsNesterov optimizer = RmsNesterov().with_momentum(0.9);
	RmsProp optimizer = RmsProp();
	
	optimizer.initialise(100, autoencoder);
	optimizer.autoencode(metadata_set, 40);
	
	autoencoder.save("2Dcompress");
	Matrix outputs;
	metadata_set = Extractor::get_metadata(file_path);
	outputs = autoencoder.get_outputs_in_layer(2, metadata_set);
	Extractor::save_matrix("Backups/Compressed_outputs", outputs);
	Displayer outputs_displayer("OUTPUTS 2D PLOT", 0, 0, MODE_2D_PLOT);
	outputs_displayer.draw_2d(outputs.fetch());
	
	*/
	
	
	/*
	
	//Clustering
	DataBlock outputs = Extractor::load_matrix("Backups/Compressed_outputs").fetch();
	Clustering cluster_maker(outputs, 500, 5);
	DataBlock clusters = cluster_maker.get_clusters(50);
	Displayer clusters_displayer("OUTPUTS CLUSTERING", 0, 0, MODE_CLUSTERING);
	clusters_displayer.draw_clusters(outputs, clusters);
	
	*/
	return 0;

} 

/*
catch (const std::exception &error) {
 std::cerr << "Deep Learning Error : " << error.what() << std::endl;
 throw (error);
 return -1;
 
}
*/



