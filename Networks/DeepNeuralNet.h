//
//  DeepNeuralNet.h
//  deep learning
//
//  Created by Sebastien Raffray on 04/07/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__DeepNeuralNet__
#define __deep_learning__DeepNeuralNet__

#include "Network.h"

// A Deep Neural Network is a Network which is terminated by a loss function
// To construct a DNN, you need to provide :
// - a vector of Layers
// - a loss function


class DeepNeuralNet : public Network {
public:
	DeepNeuralNet(vector<Layer*> layers, const Layer *loss, int mode);
	virtual ~DeepNeuralNet();
	
};

#endif /* defined(__deep_learning__DeepNeuralNet__) */
