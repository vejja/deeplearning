//
//  Extractor.cpp
//  deep learning
//
//  Created by Sebastien Raffray on 13/04/2015.
//  Copyright (c) 2015 Vejja. All rights reserved.
//

#include "Extractor.h"

Matrix	Extractor::get_images(string filepath,
	unsigned int nb_extract)// Matrix &images)
{
	Matrix images;
	unsigned int	big_endian_tmp;
	
	streampos		size;
	unsigned int	magic_number;
	unsigned int	nb_images_in_file;
	unsigned int	image_height;
	unsigned int	image_width;
	unsigned int	nb_elements;
	unsigned char	*memblock;
	
	unsigned int	nbr_images;
	unsigned int	pix_per_img;
	DataBlock			temp_images;
	
	ifstream file(filepath, ios::in | ios::binary | ios::ate);
	if (!file.is_open()) throw std::runtime_error("Extractor - Unable to open file");

	size = file.tellg();
	file.seekg(0, ios::beg);
		
	// Do we have the magic number 2051 ?
	file.read((char *)&big_endian_tmp, 4);
	magic_number = ntohl(big_endian_tmp);
	if (magic_number != 2051) {
		file.close();
		throw std::runtime_error("Extractor - wrong file signature");
	}
		
	// check total nbr of images in file
	file.read((char *)&big_endian_tmp, 4);
	nb_images_in_file = ntohl(big_endian_tmp);
	cout << "Nb of images in file : " <<nb_images_in_file << endl;
	if (nb_extract > nb_images_in_file) throw std::runtime_error("Extractor - Request for wrong number of images");
		
	// sets nbr of images to extract
	nbr_images = nb_extract;
		
	cout << "Extracting " << nbr_images << " images from file" << endl;
	// nbr of rows per image
	file.read((char *)&big_endian_tmp, 4);
	image_height = ntohl(big_endian_tmp);
		
	// nbr of columns per image
	file.read((char *)&big_endian_tmp, 4);
	image_width = ntohl(big_endian_tmp);
		
	// pixels per image
	pix_per_img = image_height * image_width;
		
	temp_images = DataBlock(nbr_images, pix_per_img);
		
	// total number of elements to read from file
	nb_elements = pix_per_img * nbr_images;
		
	// read all uint values in one block
	memblock = new unsigned char[nb_elements];
	file.read((char *)memblock, nb_elements);
	file.close();
		
	// convert uint values to floats and store in vector
	for (unsigned int i = 0; i < nb_elements; i++) {
		temp_images.cpu_buffer[i] = (float)memblock[i];
	}
		
	// release the block
	delete[] memblock;
	temp_images.insert_left_column(1);
	images = Matrix(temp_images);
	return images;
}

Matrix	Extractor::get_labels(string filepath,
							  unsigned int nb_extract)
{
	Matrix images;
	unsigned int	big_endian_tmp;
	
	streampos		size;
	unsigned int	magic_number;
	unsigned int	nb_images_in_file;
	unsigned int	nb_elements;
	unsigned char	*memblock;
	
	unsigned int	nbr_images;
	unsigned int	pix_per_img;
	DataBlock			temp_images;
	
	ifstream file(filepath, ios::in | ios::binary | ios::ate);
	if (!file.is_open()) throw std::runtime_error("Extractor - Unable to open file");

	size = file.tellg();
	file.seekg(0, ios::beg);
		
	// Do we have the magic number 2049 ?
	file.read((char *)&big_endian_tmp, 4);
	magic_number = ntohl(big_endian_tmp);
	if (magic_number != 2049) {
		file.close();
		throw std::runtime_error("Extractor - wrong file signature");
	}
		
	// check total nbr of images in file
	file.read((char *)&big_endian_tmp, 4);
	nb_images_in_file = ntohl(big_endian_tmp);
	cout << "Nb of images in file : " <<nb_images_in_file << endl;
	if (nb_extract > nb_images_in_file) throw std::runtime_error("Extractor - Request for wrong number of images");
		
	// sets nbr of images to extract
	nbr_images = nb_extract;
		
	cout << "Extracting " << nbr_images << " labels from file" << endl;
	pix_per_img = 10;
	temp_images = DataBlock(nbr_images, pix_per_img);
		
	nb_elements = nbr_images;
	memblock = new unsigned char [nb_elements];
		
	file.read((char *)memblock, nb_elements);
	file.close();
	for (unsigned int i = 0; i < nb_elements; i++) {
		for (unsigned int j = 0; j < 10; j++) {
			if ((unsigned int)memblock[i] == j)
				temp_images.cpu_buffer[i*10 + j] = 1;
			else
				temp_images.cpu_buffer[i*10 + j] = 0;
		}
	}
		
	delete[] memblock;
	temp_images.insert_left_column(1);
	images = Matrix(temp_images);
	return images;

}

void	Extractor::scale(Matrix &images, float scalar)
{
	images *= scalar;
	images.fill_left_column(1);
}

void	Extractor::shift(Matrix &images, float scalar)
{
	images += scalar;
	images.fill_left_column(1);
}

Matrix	Extractor::get_metadata(string filepath)
{
	Matrix result;
	
	streampos		file_size;
	streampos		header_size;

	unsigned int	total_bytes;
	char			*header;
	unsigned char	*memblock;
	char			single_byte;

	DataBlock		temp_images;
	
	ifstream file(filepath, ios::in | ios::binary | ios::ate);
	if (!file.is_open()) throw std::runtime_error("Extractor - Unable to open file");
	
	file.seekg(0, ios::end);
	file_size = file.tellg();
	file.seekg(0, ios::beg);
	
	bool must_continue = true;
	while (must_continue) {
		file.read(&single_byte, 1);
		if (strncmp(&single_byte, "\0", 1) == 0) {
			header_size = file.tellg();
			must_continue = false;
		}
	}
	
	header = new char[header_size];
	file.seekg(0, ios::beg);
	file.read(header, (int)header_size);
	
	string header_string(header);
	delete[] header;
	
	size_t total_pos = header_string.find("TOTAL = ");
	string length_str = header_string.substr(total_pos + 8, header_string.length() - (total_pos + 8) - 1);
	unsigned int record_length = stoi(length_str);
	
	total_bytes = (unsigned int)file_size - (unsigned int)header_size;
	unsigned int nbr_records = total_bytes / record_length;
	if (total_bytes % record_length != 0) throw std::runtime_error("Extractor - File size does not match parameters");
	
	// read all bytes in one block
	memblock = new unsigned char[total_bytes];
	file.read((char *)memblock, total_bytes);
	file.close();
	
	// convert uint values to floats and store in vector
	temp_images = DataBlock(nbr_records, record_length);
	for (unsigned int i = 0; i < total_bytes; i++) {
		temp_images.cpu_buffer[i] = (float)memblock[i];
	}


	// release the block
	delete[] memblock;
	temp_images.insert_left_column(1);
	/*
	for (unsigned int i = 0; i < temp_images.nb_rows; i++) {
		string rep;
		for (unsigned int j = 0; j<temp_images.nb_cols; j++) {
			unsigned char value = (unsigned char)temp_images.cpu_buffer[i * temp_images.nb_cols + j];
			rep.append((char *)&value, 1);
		}
		cout << rep << endl << endl;
	}
 */
	
	result = Matrix(temp_images);
	return result;
}

void	Extractor::standardize(Matrix &data)
{
	data.standardize();
	data.fill_left_column(1);
}

Matrix	Extractor::load_matrix(string filepath)
{
	Matrix result;
	
	streampos			size;
	unsigned long int	total_bytes;
	unsigned int		nb_rows;
	unsigned int		nb_cols;
	DataBlock			temp_images;
	
	ifstream file(filepath, ios::in | ios::binary | ios::ate);
	if (!file.is_open()) throw std::runtime_error("Extractor - Unable to open file");
	
	file.seekg(0, ios::end);
	size = file.tellg();
	file.seekg(0, ios::beg);
	
	
	// read rows and columns params
	file.read((char *)&nb_rows, sizeof(nb_rows));
	file.read((char *)&nb_cols, sizeof(nb_cols));
	
	// read all bytes in one block
	total_bytes = nb_rows * nb_cols * sizeof(float);
	if (total_bytes + sizeof(nb_rows) + sizeof(nb_cols) != (int)size) throw std::runtime_error("Extractor - File size does not match parameters");
	temp_images = DataBlock(nb_rows, nb_cols);
	file.read((char *)temp_images.cpu_buffer, total_bytes);
	file.close();
	
	result = Matrix(temp_images);
	return result;
}

void	Extractor::save_matrix(string filepath, const Matrix &data)
{
	unsigned long int	total_bytes;
	DataBlock			temp_images = data.fetch();
	
	ofstream file(filepath, ios::out | ios::binary | ios::trunc);
	if (!file.is_open()) throw std::runtime_error("Extractor - Unable to open file");
	
	// write rows and columns
	file.write((char *)&temp_images.nb_rows, sizeof(temp_images.nb_rows));
	file.write((char *)&temp_images.nb_cols, sizeof(temp_images.nb_cols));
	
	
	// write all bytes in one block
	total_bytes = data.nb_elements() * sizeof(float);
	file.write((char *)temp_images.cpu_buffer, total_bytes);
	file.close();
}

bool	Extractor::network_exists(string name)
{
	streampos size;
	ifstream file("Backups/" + name + ".dln", ios::in | ios::binary | ios::ate);
	if (!file.is_open()) return false;
	
	file.seekg(0, ios::end);
	size = file.tellg();
	file.close();
	return (size > 0);
}


