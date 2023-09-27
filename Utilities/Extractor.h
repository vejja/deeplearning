//
//  Extractor.h
//  deep learning
//
//  Created by Sebastien Raffray on 13/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Extractor__
#define __deep_learning__Extractor__

#include "../GPU Acceleration/Matrix.h"

class Extractor
{
public:

	static Matrix	get_images(string filepath, unsigned int nb_extract);
	static Matrix	get_labels(string filepath, unsigned int nb_extract);
	static void		scale(Matrix &images, float scalar);
	static void		shift(Matrix &images, float scalar);
	static Matrix	get_metadata(string filepath);
	static void		standardize(Matrix &data);
	static Matrix	load_matrix(string filepath);
	static void		save_matrix(string filepath, const Matrix &data);
	static bool		network_exists(string name);
};

#endif /* defined(__deep_learning__Extractor__) */
