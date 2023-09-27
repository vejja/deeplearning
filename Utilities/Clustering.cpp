//
//  Clustering.cpp
//  deep learning
//
//  Created by Fernando Raffray on 09/07/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "Clustering.h"

Object::Object()
{
}

Object::~Object()
{
}

float Object::distance(const Object &other_object) const
{
	float distance = 0;
	for (unsigned int i = 0; i < values.size(); i++) {
		distance += powf(values[i] - other_object.values[i], 2);
	}
	return sqrt(distance);
}

multimap<float, Object*> Object::neighbors(vector<Object> &objects, float epsilon) const
{
	multimap<float, Object*> neighbors;
	
	for (auto &object : objects) {
		const float dist = distance(object);
		if (dist < epsilon) {
			neighbors.insert(make_pair(dist, &object));
		}
		//else cout << i << endl;
	}
	return neighbors;
}


void Object::set_core_distance(const multimap<float, Object*> &neighbors, float epsilon, unsigned int min_points)
{
	if (neighbors.size() < min_points) {
		core_distance = DIST_UNDEFINED;
	}
	else {
		auto iterator = neighbors.begin();
		std::advance(iterator, min_points - 1);
		core_distance = iterator->first;
	}
}

Clustering::Clustering(const DataBlock &points, float epsilon, unsigned int min_points)
{
	param_epsilon = epsilon;
	param_min_points = min_points;
	for (unsigned int row = 0; row < points.nb_rows; row++) {
		Object new_object;
		for (unsigned int col = 1; col < points.nb_cols; col++) {
			new_object.values.push_back(points.cpu_buffer[row * points.nb_cols + col]);
		}
		set_of_objects.push_back(new_object);
	}
}

Clustering::~Clustering()
{
}

DataBlock Clustering::get_clusters(float epsilon_prime) {
	
	DataBlock clusters((unsigned int)set_of_objects.size(), 1);
	optics();
	extract_dbscan_clustering(epsilon_prime);
	
	for (unsigned int i = 0; i < set_of_objects.size(); i++) {
		clusters.cpu_buffer[i] = set_of_objects[i].cluster_id;
	}
	return clusters;
}

void Clustering::optics()
 /*
 OPTICS (SetOfObjects, ε, MinPts, OrderedFile)
 OrderedFile.open();
 FOR i FROM 1 TO SetOfObjects.size DO
 Object := SetOfObjects.get(i);
 IF NOT Object.Processed THEN
 ExpandClusterOrder(SetOfObjects, Object, ε, MinPts, OrderedFile)
 OrderedFile.close(); END; // OPTICS
 */
{
	for (auto &object : set_of_objects) {
		if (!object.processed) expand_cluster_order(object);
	}
}

void Clustering::expand_cluster_order(Object &object)
/*
 ExpandClusterOrder(SetOfObjects, Object, ε, MinPts, OrderedFile);
 neighbors := SetOfObjects.neighbors(Object, ε); Object.Processed := TRUE; Object.reachability_distance := UNDEFINED; Object.setCoreDistance(neighbors, ε, MinPts); OrderedFile.write(Object);
 IF Object.core_distance <> UNDEFINED THEN OrderSeeds.update(neighbors, Object); WHILE NOT OrderSeeds.empty() DO
 currentObject := OrderSeeds.next(); neighbors:=SetOfObjects.neighbors(currentObject, ε); currentObject.Processed := TRUE; currentObject.setCoreDistance(neighbors, ε, MinPts); OrderedFile.write(currentObject);
 IF currentObject.core_distance<>UNDEFINED THEN OrderSeeds.update(neighbors, currentObject);
 END; // ExpandClusterOrder
 */
{
	multimap<float, Object*> neighbors = object.neighbors(set_of_objects, param_epsilon);
	object.processed = true;
	object.reachability_distance = DIST_UNDEFINED;
	object.set_core_distance(neighbors, param_epsilon, param_min_points);
	ordered_file.push_back(&object);
	if (object.core_distance != DIST_UNDEFINED) {
		update(neighbors, object);
		while (order_seeds.size() > 0) {
			Object &current_object = *order_seeds.begin()->second;
			order_seeds.erase(order_seeds.begin());
			multimap<float, Object*> new_neighbors = current_object.neighbors(set_of_objects, param_epsilon);
			current_object.processed = true;
			current_object.set_core_distance(new_neighbors, param_epsilon, param_min_points);
			ordered_file.push_back(&current_object);
			if (current_object.core_distance != DIST_UNDEFINED) {
				update(new_neighbors, current_object);
			}
		}
	}
}

void Clustering::update(const multimap<float, Object*> &neighbors, const Object &center_object)
/*
 OrderSeeds::update(neighbors, CenterObject); c_dist := CenterObject.core_distance; FORALL Object FROM neighbors DO
 IF NOT Object.Processed THEN new_r_dist:=max(c_dist,CenterObject.dist(Object)); IF Object.reachability_distance=UNDEFINED THEN
 Object.reachability_distance := new_r_dist;
 insert(Object, new_r_dist);
 ELSE // Object already in OrderSeeds
 IF new_r_dist<Object.reachability_distance THEN Object.reachability_distance := new_r_dist; decrease(Object, new_r_dist);
 END; // OrderSeeds::update
 */
{
	float core_dist = center_object.core_distance;
	//cout << "size : " << neighbors.size() << endl;
	for (auto &neighbor : neighbors) {
		Object &object = *neighbor.second;
		if (!object.processed) {
			float new_reach_dist = fmaxf(core_dist, neighbor.first);
			//if (new_reach_dist == core_dist) cout << "objet trouvé à " << center_object.distance(object) << endl;
			if (object.reachability_distance == DIST_UNDEFINED) { // object is not in seeds
				object.reachability_distance = new_reach_dist;
				order_seeds.insert(make_pair(new_reach_dist, &object));
			}
			else { // object is already in seeds, check for improvement
				if (new_reach_dist < object.reachability_distance) {
					order_seeds.erase(&object);
					object.reachability_distance = new_reach_dist;
					order_seeds.insert(&object);
				}
			}
		}
		//else cout << "already processed : " << i << endl;
	}
}

void Clustering::extract_dbscan_clustering(float epsilon_prime){
/*
 ExtractDBSCAN-Clustering (ClusterOrderedObjs,ε’, MinPts) // Precondition: ε' ≤ generating dist ε for ClusterOrderedObjs
 ClusterId := NOISE;
 FOR i FROM 1 TO ClusterOrderedObjs.size DO
 Object := ClusterOrderedObjs.get(i);
 IF Object.reachability_distance > ε’ THEN
 // UNDEFINED > ε
 IF Object.core_distance ≤ ε’ THEN
 ClusterId := nextId(ClusterId);
 Object.clusterId := ClusterId; ELSE
 Object.clusterId := NOISE;
 ELSE // Object.reachability_distance ≤ ε’
 Object.clusterId := ClusterId; END; // ExtractDBSCAN-Clustering
*/
	unsigned int current_cluster_id = 0;
	for (auto object_ptr : ordered_file) {
		if (object_ptr->reachability_distance > epsilon_prime) {
			if (object_ptr->core_distance <= epsilon_prime) {
				current_cluster_id++;
				object_ptr->cluster_id = current_cluster_id;
			}
			else {
				object_ptr->cluster_id = 0;
			}
		}
		else {
			object_ptr->cluster_id = current_cluster_id;
		}
	}
}