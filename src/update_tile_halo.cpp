/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under 
 the terms of the GNU General Public License as published by the 
 Free Software Foundation, either version 3 of the License, or (at your option) 
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but 
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */


#include "update_tile_halo.h"
#include "update_tile_halo_kernel.h"

#include <iostream>

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_tile_halo(global_variables &globals, int fields[NUM_FIELDS], int depth) {

	// Update Top Bottom - Real to Real

	for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {
		int t_up = globals.chunk.tiles[tile].tile_neighbours[tile_top];
		int t_down = globals.chunk.tiles[tile].tile_neighbours[tile_bottom];

		if (t_up != external_tile) {
			update_tile_halo_t_kernel(
					globals.chunk.tiles[tile].t_xmin,
					globals.chunk.tiles[tile].t_xmax,
					globals.chunk.tiles[tile].t_ymin,
					globals.chunk.tiles[tile].t_ymax,
					globals.chunk.tiles[tile].field.density0,
					globals.chunk.tiles[tile].field.energy0,
					globals.chunk.tiles[tile].field.pressure,
					globals.chunk.tiles[tile].field.viscosity,
					globals.chunk.tiles[tile].field.soundspeed,
					globals.chunk.tiles[tile].field.density1,
					globals.chunk.tiles[tile].field.energy1,
					globals.chunk.tiles[tile].field.xvel0,
					globals.chunk.tiles[tile].field.yvel0,
					globals.chunk.tiles[tile].field.xvel1,
					globals.chunk.tiles[tile].field.yvel1,
					globals.chunk.tiles[tile].field.vol_flux_x,
					globals.chunk.tiles[tile].field.vol_flux_y,
					globals.chunk.tiles[tile].field.mass_flux_x,
					globals.chunk.tiles[tile].field.mass_flux_y,
					globals.chunk.tiles[t_up].t_xmin,
					globals.chunk.tiles[t_up].t_xmax,
					globals.chunk.tiles[t_up].t_ymin,
					globals.chunk.tiles[t_up].t_ymax,
					globals.chunk.tiles[t_up].field.density0,
					globals.chunk.tiles[t_up].field.energy0,
					globals.chunk.tiles[t_up].field.pressure,
					globals.chunk.tiles[t_up].field.viscosity,
					globals.chunk.tiles[t_up].field.soundspeed,
					globals.chunk.tiles[t_up].field.density1,
					globals.chunk.tiles[t_up].field.energy1,
					globals.chunk.tiles[t_up].field.xvel0,
					globals.chunk.tiles[t_up].field.yvel0,
					globals.chunk.tiles[t_up].field.xvel1,
					globals.chunk.tiles[t_up].field.yvel1,
					globals.chunk.tiles[t_up].field.vol_flux_x,
					globals.chunk.tiles[t_up].field.vol_flux_y,
					globals.chunk.tiles[t_up].field.mass_flux_x,
					globals.chunk.tiles[t_up].field.mass_flux_y,
					fields,
					depth);

		}

		if (t_down != external_tile) {
			update_tile_halo_b_kernel(
					globals.chunk.tiles[tile].t_xmin,
					globals.chunk.tiles[tile].t_xmax,
					globals.chunk.tiles[tile].t_ymin,
					globals.chunk.tiles[tile].t_ymax,
					globals.chunk.tiles[tile].field.density0,
					globals.chunk.tiles[tile].field.energy0,
					globals.chunk.tiles[tile].field.pressure,
					globals.chunk.tiles[tile].field.viscosity,
					globals.chunk.tiles[tile].field.soundspeed,
					globals.chunk.tiles[tile].field.density1,
					globals.chunk.tiles[tile].field.energy1,
					globals.chunk.tiles[tile].field.xvel0,
					globals.chunk.tiles[tile].field.yvel0,
					globals.chunk.tiles[tile].field.xvel1,
					globals.chunk.tiles[tile].field.yvel1,
					globals.chunk.tiles[tile].field.vol_flux_x,
					globals.chunk.tiles[tile].field.vol_flux_y,
					globals.chunk.tiles[tile].field.mass_flux_x,
					globals.chunk.tiles[tile].field.mass_flux_y,
					globals.chunk.tiles[t_down].t_xmin,
					globals.chunk.tiles[t_down].t_xmax,
					globals.chunk.tiles[t_down].t_ymin,
					globals.chunk.tiles[t_down].t_ymax,
					globals.chunk.tiles[t_down].field.density0,
					globals.chunk.tiles[t_down].field.energy0,
					globals.chunk.tiles[t_down].field.pressure,
					globals.chunk.tiles[t_down].field.viscosity,
					globals.chunk.tiles[t_down].field.soundspeed,
					globals.chunk.tiles[t_down].field.density1,
					globals.chunk.tiles[t_down].field.energy1,
					globals.chunk.tiles[t_down].field.xvel0,
					globals.chunk.tiles[t_down].field.yvel0,
					globals.chunk.tiles[t_down].field.xvel1,
					globals.chunk.tiles[t_down].field.yvel1,
					globals.chunk.tiles[t_down].field.vol_flux_x,
					globals.chunk.tiles[t_down].field.vol_flux_y,
					globals.chunk.tiles[t_down].field.mass_flux_x,
					globals.chunk.tiles[t_down].field.mass_flux_y,
					fields,
					depth);
		}
	}


	// Update Left Right - Ghost, Real, Ghost - > Real

	for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {
		int t_left = globals.chunk.tiles[tile].tile_neighbours[tile_left];
		int t_right = globals.chunk.tiles[tile].tile_neighbours[tile_right];

		if (t_left != external_tile) {
			update_tile_halo_l_kernel(
					globals.chunk.tiles[tile].t_xmin,
					globals.chunk.tiles[tile].t_xmax,
					globals.chunk.tiles[tile].t_ymin,
					globals.chunk.tiles[tile].t_ymax,
					globals.chunk.tiles[tile].field.density0,
					globals.chunk.tiles[tile].field.energy0,
					globals.chunk.tiles[tile].field.pressure,
					globals.chunk.tiles[tile].field.viscosity,
					globals.chunk.tiles[tile].field.soundspeed,
					globals.chunk.tiles[tile].field.density1,
					globals.chunk.tiles[tile].field.energy1,
					globals.chunk.tiles[tile].field.xvel0,
					globals.chunk.tiles[tile].field.yvel0,
					globals.chunk.tiles[tile].field.xvel1,
					globals.chunk.tiles[tile].field.yvel1,
					globals.chunk.tiles[tile].field.vol_flux_x,
					globals.chunk.tiles[tile].field.vol_flux_y,
					globals.chunk.tiles[tile].field.mass_flux_x,
					globals.chunk.tiles[tile].field.mass_flux_y,
					globals.chunk.tiles[t_left].t_xmin,
					globals.chunk.tiles[t_left].t_xmax,
					globals.chunk.tiles[t_left].t_ymin,
					globals.chunk.tiles[t_left].t_ymax,
					globals.chunk.tiles[t_left].field.density0,
					globals.chunk.tiles[t_left].field.energy0,
					globals.chunk.tiles[t_left].field.pressure,
					globals.chunk.tiles[t_left].field.viscosity,
					globals.chunk.tiles[t_left].field.soundspeed,
					globals.chunk.tiles[t_left].field.density1,
					globals.chunk.tiles[t_left].field.energy1,
					globals.chunk.tiles[t_left].field.xvel0,
					globals.chunk.tiles[t_left].field.yvel0,
					globals.chunk.tiles[t_left].field.xvel1,
					globals.chunk.tiles[t_left].field.yvel1,
					globals.chunk.tiles[t_left].field.vol_flux_x,
					globals.chunk.tiles[t_left].field.vol_flux_y,
					globals.chunk.tiles[t_left].field.mass_flux_x,
					globals.chunk.tiles[t_left].field.mass_flux_y,
					fields,
					depth);
		}

		if (t_right != external_tile) {
			update_tile_halo_r_kernel(
					globals.chunk.tiles[tile].t_xmin,
					globals.chunk.tiles[tile].t_xmax,
					globals.chunk.tiles[tile].t_ymin,
					globals.chunk.tiles[tile].t_ymax,
					globals.chunk.tiles[tile].field.density0,
					globals.chunk.tiles[tile].field.energy0,
					globals.chunk.tiles[tile].field.pressure,
					globals.chunk.tiles[tile].field.viscosity,
					globals.chunk.tiles[tile].field.soundspeed,
					globals.chunk.tiles[tile].field.density1,
					globals.chunk.tiles[tile].field.energy1,
					globals.chunk.tiles[tile].field.xvel0,
					globals.chunk.tiles[tile].field.yvel0,
					globals.chunk.tiles[tile].field.xvel1,
					globals.chunk.tiles[tile].field.yvel1,
					globals.chunk.tiles[tile].field.vol_flux_x,
					globals.chunk.tiles[tile].field.vol_flux_y,
					globals.chunk.tiles[tile].field.mass_flux_x,
					globals.chunk.tiles[tile].field.mass_flux_y,
					globals.chunk.tiles[t_right].t_xmin,
					globals.chunk.tiles[t_right].t_xmax,
					globals.chunk.tiles[t_right].t_ymin,
					globals.chunk.tiles[t_right].t_ymax,
					globals.chunk.tiles[t_right].field.density0,
					globals.chunk.tiles[t_right].field.energy0,
					globals.chunk.tiles[t_right].field.pressure,
					globals.chunk.tiles[t_right].field.viscosity,
					globals.chunk.tiles[t_right].field.soundspeed,
					globals.chunk.tiles[t_right].field.density1,
					globals.chunk.tiles[t_right].field.energy1,
					globals.chunk.tiles[t_right].field.xvel0,
					globals.chunk.tiles[t_right].field.yvel0,
					globals.chunk.tiles[t_right].field.xvel1,
					globals.chunk.tiles[t_right].field.yvel1,
					globals.chunk.tiles[t_right].field.vol_flux_x,
					globals.chunk.tiles[t_right].field.vol_flux_y,
					globals.chunk.tiles[t_right].field.mass_flux_x,
					globals.chunk.tiles[t_right].field.mass_flux_y,
					fields,
					depth);
		}
	}
}

