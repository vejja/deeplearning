//
//  Clustering.h
//  deep learning
//
//  Created by Fernando Raffray on 09/07/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Clustering__
#define __deep_learning__Clustering__

#include "DataBlock.h"
#include <vector>
#include <set>
#include <map>

#define DIST_UNDEFINED MAXFLOAT

// This class implements the OPTICS clustering algorithm
// By Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jörg Sander - 1999

class Object {
public:
	Object();
	~Object();
	
	vector<float> values;
	bool processed = false;
	//bool reachability_distance_defined = false;
	float reachability_distance = DIST_UNDEFINED;
	//bool core_distance_defined = false;
	float core_distance = DIST_UNDEFINED;
	unsigned int cluster_id;
	
	float distance(const Object &other_object) const;
	
	multimap<float, Object*> neighbors(vector<Object> &objects, float epsilon) const;
	void set_core_distance(const multimap<float, Object*> &neighbors, float epsilon, unsigned int min_points);
	
	//bool closest(Object *lhs, Object* rhs);
	
};
/*
struct ObjectPtrReachCompare {
	bool operator() (const Object *lhs, const Object *rhs) const {
		return (lhs->reachability_distance < rhs->reachability_distance);
	}
};
*/

class Clustering {
public:
	Clustering(const DataBlock &points, float epsilon, unsigned int min_points);
	~Clustering();

	DataBlock get_clusters(float epsilon_prime);

private:
	float param_epsilon;
	unsigned int param_min_points;
	vector<Object> set_of_objects;
	vector<Object*> ordered_file;
	multimap<float, Object*> order_seeds;
	
	void optics();
	void update(const multimap<float, Object*> &neighbors, const Object &center_object);
	void expand_cluster_order(Object &object);
	void extract_dbscan_clustering(float epsilon_prime);
};

#endif /* defined(__deep_learning__Clustering__) */
