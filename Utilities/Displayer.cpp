//
//  Displayer.cpp
//  deep learning
//
//  Created by Sébastien Raffray on 27/01/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#include "Displayer.h"

int Displayer::nbr_windows = 0;
bool Displayer::exit_requested = false;

Displayer::Displayer(string title, unsigned int nbr_inputs, unsigned int nbr_outputs, int mode)
{
	int window_width;
	int window_height;
	int window_result;

	
	setup();
    window_title = title;
	window_mode = mode;
	
	if (window_mode == MODE_2D_PLOT || window_mode == MODE_CLUSTERING) {
		window_width = 600;
		window_height = 600;
		left_tile_side = 0;
		right_tile_side = 0;
	}
	
	else {
		left_tile_side = (int)sqrt(nbr_inputs);
		if (window_mode == MODE_INPUTS_AND_WEIGHTS) {
			right_tile_side = left_tile_side;
		}
		else if (window_mode == MODE_INPUTS_AND_OUTPUTS) {
			right_tile_side = (int)sqrt(nbr_outputs);
		}
		else throw std::runtime_error(" Displayer - Displayer mode not recognized");
		
		window_width = (1 + left_tile_side) * 10 + 5 + (1 + right_tile_side) * 10;
		window_height = (right_tile_side > left_tile_side) ? (1 + right_tile_side) * 10 : (1 + left_tile_side) * 10;
	}
	// Création de la fenêtre :
	window_result = SDL_CreateWindowAndRenderer(window_width, window_height, SDL_WINDOW_SHOWN, &window, &renderer);

	if (window_result < 0)
	{
		cout << "Erreur lors de la creation d'un renderer : " << SDL_GetError() << endl;
		throw std::runtime_error("Displayer - SDL Error");
	}
	
	SDL_SetWindowTitle(window, window_title.c_str());
	SDL_SetWindowPosition(window, nbr_windows*100, SDL_WINDOWPOS_CENTERED);
	clear_window();
	display_window();
}

Displayer::~Displayer() {
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	cleanup();
}


string      Displayer::get_title() const
{
    return window_title;
}

int			Displayer::get_left_tile_side() const
{
	return left_tile_side;
}

int			Displayer::get_right_tile_side() const
{
	return right_tile_side;
}


void		Displayer::draw_layer(const DataBlock &inputs, const DataBlock &weights, const DataBlock &outputs)
{
	if (window_mode == MODE_INPUTS_AND_WEIGHTS) {
		draw_inputs(inputs);
		draw_weights(weights);
	}
	else if (window_mode == MODE_INPUTS_AND_OUTPUTS) {
		draw_inputs(inputs);
		draw_outputs(outputs);
	}
	else throw std::runtime_error("Displayer - Displayer mode not recognized");
}

void		Displayer::draw_2d(const DataBlock &points)
{
	float value;
	if (points.nb_cols != 3) throw std::runtime_error("Displayer - cannot plot on 2D graph if outputs are not compressed to 2D");
	
	float x_min = points.cpu_buffer[1];
	float x_max = x_min;
	for (unsigned int pos = 4; pos < points.size(); pos += 3) {
		value = points.cpu_buffer[pos];
		if (value > x_max) x_max = value;
		if (value < x_min) x_min = value;
	}
	
	float y_min = points.cpu_buffer[2];
	float y_max = y_min;
	for (unsigned int pos = 5; pos < points.size(); pos += 3) {
		value = points.cpu_buffer[pos];
		if (value > y_max) y_max = value;
		if (value < y_min) y_min = value;
	}

	x_min = abs(x_min);
	x_max = abs(x_max);
	y_min = abs(y_min);
	y_max = abs(y_max);
	x_max = (x_max > x_min) ? x_max : x_min;
	y_max = (y_max > y_min) ? y_max : y_min;
	
	for (unsigned int pos = 1; pos < points.size(); pos +=3) {
		put_pixel(300 + 300 * points.cpu_buffer[pos] / x_max, 300 + 300 * points.cpu_buffer[pos+1] / y_max, 250, 100, 200);
	}
	display_window();
}

void		Displayer::draw_clusters(const DataBlock &points, const DataBlock &clusters)
{
	float value;
	if (points.nb_cols != 3) throw std::runtime_error("Displayer - cannot plot on 2D graph if outputs are not compressed to 2D");
	if (points.nb_rows != clusters.nb_rows) throw std::runtime_error("Displayer - number of points in clusters is incorrect");
	
	float x_min = points.cpu_buffer[1];
	float x_max = x_min;
	for (unsigned int pos = 4; pos < points.size(); pos += 3) {
		value = points.cpu_buffer[pos];
		if (value > x_max) x_max = value;
		if (value < x_min) x_min = value;
	}
	
	float y_min = points.cpu_buffer[2];
	float y_max = y_min;
	for (unsigned int pos = 5; pos < points.size(); pos += 3) {
		value = points.cpu_buffer[pos];
		if (value > y_max) y_max = value;
		if (value < y_min) y_min = value;
	}
	
	x_min = abs(x_min);
	x_max = abs(x_max);
	y_min = abs(y_min);
	y_max = abs(y_max);
	x_max = (x_max > x_min) ? x_max : x_min;
	y_max = (y_max > y_min) ? y_max : y_min;
	
	struct rgb_color {
		uint8_t red;
		uint8_t green;
		uint8_t blue;
	};
	
	vector <rgb_color> palette;

	for (unsigned int row = 0; row < points.nb_rows; row++) {
		unsigned int x_pos = row * 3 + 1;
		unsigned int y_pos = row * 3 + 2;
		unsigned int cluster_id = clusters.cpu_buffer[row];
		rgb_color cluster_color;
		
		while (cluster_id >= palette.size()) {
			cluster_color.red = rand() % 256;
			cluster_color.green = rand() % 256;
			cluster_color.blue = rand() % 256;
			palette.push_back(cluster_color);
		}
		cluster_color = palette[cluster_id];
		put_pixel(300 + 300 * points.cpu_buffer[x_pos] / x_max, 300 + 300 * points.cpu_buffer[y_pos] / y_max, cluster_color.red, cluster_color.green, cluster_color.blue);
	}
	display_window();
}


void		Displayer::setup()
{
	if (nbr_windows == 0) {
		SDL_Init(SDL_INIT_VIDEO); // Initialisation de la SDL
	}
	nbr_windows++;
}

DataBlock		Displayer::normalise(const DataBlock &inputs)
{
	DataBlock normalised_inputs = inputs;
	float min_value = 0;
	float max_value = 1;
	
	for (unsigned int i = 0; i < inputs.size(); i++) {
		if (inputs.cpu_buffer[i] > max_value) {
			max_value = inputs.cpu_buffer[i];
		}
		if (inputs.cpu_buffer[i] < min_value) {
			min_value = inputs.cpu_buffer[i];
		}
	}
	
	normalised_inputs.add_scalar(-min_value);
	normalised_inputs.multiply_by_scalar(255.0f / (max_value - min_value));
	
	std::cout << "Normalization done for range [" << min_value << " - " << max_value << "]" << std::endl;
	return normalised_inputs;
}

void		Displayer::draw_inputs(const DataBlock &inputs)
{
	int x_offset, y_offset;
	DataBlock normalised_inputs;
	DataBlock line;
	
	normalised_inputs = normalise(inputs);
	normalised_inputs.strip_left_column();
	
	for (int img_nbr = 0; img_nbr < inputs.nb_rows; img_nbr++) {
		line = normalised_inputs.extract_row(img_nbr);
		x_offset = (img_nbr % 10) * (1 + left_tile_side);
		y_offset = (img_nbr / 10) * (1 + left_tile_side);
		draw_tile(x_offset, y_offset, line);
	}
}


void		Displayer::draw_weights(const DataBlock &weights)
{
	int x_offset, y_offset;
	DataBlock normalised_weights;
	DataBlock line;
	
	normalised_weights = normalise(weights);
	normalised_weights.strip_left_column();
	
	for (int neur_col = 0; neur_col < (weights.nb_cols < 100 ? weights.nb_cols : 100); neur_col++) {
		line = normalised_weights.extract_column(neur_col);
		x_offset = (1 + left_tile_side) * 10 + 5 + (neur_col % 10) * (1 + right_tile_side);
		y_offset = (neur_col / 10) * (1 + right_tile_side);
		draw_tile(x_offset, y_offset, line);
	}
}

void		Displayer::draw_outputs(const DataBlock &outputs)
{
	int x_offset, y_offset;
	DataBlock normalised_outputs;
	DataBlock line;
	
	normalised_outputs = normalise(outputs);
	normalised_outputs.strip_left_column();
	
	for (int img_nbr = 0; img_nbr < outputs.nb_rows; img_nbr++) {
		line = normalised_outputs.extract_row(img_nbr);
		x_offset = (1 + left_tile_side) * 10 + 5 + (img_nbr % 10) * (1 + right_tile_side);
		y_offset = (img_nbr / 10) * (1 + right_tile_side);
		draw_tile(x_offset, y_offset, line);
	}
}

void		Displayer::draw_tile(int x_offset, int y_offset, const DataBlock &line)
{
	int tile_side = (int)sqrt(line.nb_cols);
	int x, y;
	uint8_t grey_color;
	
	for (int pos = 0; pos < line.nb_cols; pos++) {
		x = x_offset + (pos % tile_side);
		y = y_offset + (pos / tile_side);
		grey_color = (uint8_t)(line.cpu_buffer[pos]);
		put_pixel(x, y, grey_color, grey_color, grey_color);
	}
}

void		Displayer::put_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
	
	SDL_SetRenderDrawColor(renderer, r, g, b, 0xFF);
	SDL_RenderDrawPoint(renderer, x, y);
}


void		Displayer::cleanup()
{
	nbr_windows--;
	if (nbr_windows == 0) {
		SDL_Quit();
	}
}

void		Displayer::clear_window()
{
	SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, SDL_ALPHA_OPAQUE); // noir, opacité max
	SDL_RenderClear(renderer);
}

void		Displayer::display_window()
{

	SDL_Event dummy_event;

	while (SDL_PollEvent(&dummy_event))  {
		
		//cout << "Window " << dummy_event.window.windowID << " captured event #" << dummy_event.type << " (";
		switch (dummy_event.type) {
			/*
		case SDL_FIRSTEVENT:
			cout << "do not remove(unused)";
			break;
		case SDL_QUIT:
			cout << "user - requested quit";
			break;
			*/
		case SDL_WINDOWEVENT:
			//cout << "window state change";
			switch (dummy_event.window.event) {
				/*
			case SDL_WINDOWEVENT_SHOWN:
				SDL_Log("Window %d shown", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_HIDDEN:
				SDL_Log("Window %d hidden", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_EXPOSED:
				SDL_Log("Window %d exposed", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_MOVED:
				SDL_Log("Window %d moved to %d,%d",
					dummy_event.window.windowID, dummy_event.window.data1,
					dummy_event.window.data2);
				break;
			case SDL_WINDOWEVENT_RESIZED:
				SDL_Log("Window %d resized to %dx%d",
					dummy_event.window.windowID, dummy_event.window.data1,
					dummy_event.window.data2);
				break;
			case SDL_WINDOWEVENT_MINIMIZED:
				SDL_Log("Window %d minimized", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_MAXIMIZED:
				SDL_Log("Window %d maximized", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_RESTORED:
				SDL_Log("Window %d restored", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_ENTER:
				SDL_Log("Mouse entered window %d",
					dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_LEAVE:
				SDL_Log("Mouse left window %d", dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_FOCUS_GAINED:
				SDL_Log("Window %d gained keyboard focus",
					dummy_event.window.windowID);
				break;
			case SDL_WINDOWEVENT_FOCUS_LOST:
				SDL_Log("Window %d lost keyboard focus",
					dummy_event.window.windowID);
				break;
				*/
			case SDL_WINDOWEVENT_CLOSE:
				SDL_Log("Window %d closed", dummy_event.window.windowID);
				exit_requested = true;
				break;
				/*
			default:
				SDL_Log("Window %d got unknown event %d",
					dummy_event.window.windowID, dummy_event.window.event);
				break;
				*/
			}
			break;
			/*
		case SDL_SYSWMEVENT:
			cout << "system specific event";
			break;
		case SDL_KEYDOWN:
			cout << "key pressed";
			break;
		case SDL_KEYUP:
			cout << "key released";
			break;
		case SDL_TEXTEDITING:
			cout << "keyboard text editing(composition)";
			break;
		case SDL_TEXTINPUT:
			cout << "keyboard text input";
			break;
		case SDL_MOUSEMOTION:
			cout << "mouse moved";
			break;
		case SDL_MOUSEBUTTONDOWN:
			cout << "mouse button pressed";
			break;
		case SDL_MOUSEBUTTONUP:
			cout << "mouse button released";
			break;
		case SDL_MOUSEWHEEL:
			cout << "mouse wheel motion";
			break;
		case SDL_JOYAXISMOTION:
			cout << "joystick axis motion";
			break;
		case SDL_JOYBALLMOTION:
			cout << "joystick trackball motion";
			break;
		case SDL_JOYHATMOTION:
			cout << "joystick hat position change";
			break;
		case SDL_JOYBUTTONDOWN:
			cout << "joystick button pressed";
			break;
		case SDL_JOYBUTTONUP:
			cout << "joystick button released";
			break;
		case SDL_JOYDEVICEADDED:
			cout << "joystick connected";
			break;
		case SDL_JOYDEVICEREMOVED:
			cout << "joystick disconnected";
			break;	
		case SDL_CONTROLLERAXISMOTION:
			cout << "controller axis motion";
			break;
		case SDL_CONTROLLERBUTTONDOWN:
			cout << "controller button pressed";
			break;
		case SDL_CONTROLLERBUTTONUP:
			cout << "controller button released";
			break;
		case SDL_CONTROLLERDEVICEADDED:
			cout << "controller connected";
			break;
		case SDL_CONTROLLERDEVICEREMOVED:
			cout << "controller disconnected";
			break;
		case SDL_CONTROLLERDEVICEREMAPPED:
			cout << "controller mapping updated";
			break;
		case SDL_FINGERDOWN:
			cout << "user has touched input device";
			break;
		case SDL_FINGERUP:
			cout << "user stopped touching input device";
			break;
		case SDL_FINGERMOTION:
			cout << "user is dragging finger on input device";
			break;
		
		//case SDL_DOLLARGESTURE:
		//case SDL_DOLLARRECORD:
		//case SDL_MULTIGESTURE:
		
		case SDL_CLIPBOARDUPDATE:
			cout << "the clipboard changed";
			break;		
		case SDL_DROPFILE:
			cout << "the system requests a file open";
			break;
		
		//case SDL_AUDIODEVICEADDED:
		//	cout << "a new audio device is available(>= SDL 2.0.4)";
		//	break;
		//case SDL_AUDIODEVICEREMOVED:
		//	cout << "an audio device has been removed(>= SDL 2.0.4)";
		//	break;
			
		case SDL_RENDER_TARGETS_RESET:
			cout << "the render targets have been reset and their contents need to be updated(>= SDL 2.0.2)"; 
			break;
		
		//case SDL_RENDER_DEVICE_RESET:
		//	cout << "the device has been reset and all textures need to be recreated(>= SDL 2.0.4)";
		//	break;
			
		case SDL_USEREVENT:
			cout << "a user - specified event";
			break;
		case SDL_LASTEVENT:
			cout << "only for bounding internal arrays";
			break;
		default:
			break;*/
		}
		//cout << ")" << endl; 
	}
	SDL_RenderPresent(renderer);
}


