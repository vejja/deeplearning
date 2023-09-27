//
//  Displayer.h
//  deep learning
//
//  Created by Sebastien Raffray on 27/01/2015.
//  Copyright (c) 2014-2015 Vejja. All rights reserved.
//

#ifndef __deep_learning__Displayer__
#define __deep_learning__Displayer__

#include "DataBlock.h"
//#include <algorithm> // for use of std::min
#include <SDL.h>

#define MODE_NO_DISPLAY			0 // not drawings
#define MODE_INPUTS_AND_WEIGHTS 1 // inputs and weights are drawn on the same window
#define MODE_INPUTS_AND_OUTPUTS 2 // inputs and outputs are drawn on each window
#define MODE_MIXED				3 // inputs and weights on each window, except for last one : outputs instead of weights
#define MODE_2D_PLOT			4 // plots outputs when compressed in 2D
#define MODE_CLUSTERING			5 // plots outputs grouped by clusters

class Displayer {

public:
	static bool		exit_requested;

	Displayer(string title, unsigned int nbr_inputs, unsigned int nbr_outputs, int mode);
	~Displayer(void);

	void			set_mode(int display_mode);
    string          get_title() const;
	int				get_left_tile_side() const;
	int				get_right_tile_side() const;
	void			draw_layer(const DataBlock &inputs, const DataBlock &weights, const DataBlock &outputs);
	void			draw_2d(const DataBlock &points);
	void			draw_clusters(const DataBlock &points, const DataBlock &clusters);
	void			clear_window();
	void			display_window();

private:

	static int		nbr_windows;
	int				window_mode;
	string          window_title;
	int             left_tile_side;
	int				right_tile_side;

	SDL_Window		*window;
	SDL_Renderer	*renderer;
	
	void			setup();
	//
public:
	//
	DataBlock		normalise(const DataBlock &inputs);
	void			draw_inputs(const DataBlock &inputs);
	void			draw_weights(const DataBlock &weights);
	void			draw_outputs(const DataBlock &outputs);
	void			draw_tile(int x_offset, int y_offset, const DataBlock &line);
	void			put_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
	void			cleanup();
};

#endif /* defined(__deep_learning__Displayer__) */
