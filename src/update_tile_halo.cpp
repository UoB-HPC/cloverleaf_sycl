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
	execute(globals.queue, [&](handler &h) {

		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &tt = globals.chunk.tiles[tile];
			int t_up = tt.info.tile_neighbours[tile_top];
			int t_down = tt.info.tile_neighbours[tile_bottom];

			if (t_up != external_tile) {
				tile_type &tup = globals.chunk.tiles[t_up];
				update_tile_halo_t_kernel(
						h,
						tt.info.t_xmin,
						tt.info.t_xmax,
						tt.info.t_ymin,
						tt.info.t_ymax,
						tt.field.density0.access<RW>(h),
						tt.field.energy0.access<RW>(h),
						tt.field.pressure.access<RW>(h),
						tt.field.viscosity.access<RW>(h),
						tt.field.soundspeed.access<RW>(h),
						tt.field.density1.access<RW>(h),
						tt.field.energy1.access<RW>(h),
						tt.field.xvel0.access<RW>(h),
						tt.field.yvel0.access<RW>(h),
						tt.field.xvel1.access<RW>(h),
						tt.field.yvel1.access<RW>(h),
						tt.field.vol_flux_x.access<RW>(h),
						tt.field.vol_flux_y.access<RW>(h),
						tt.field.mass_flux_x.access<RW>(h),
						tt.field.mass_flux_y.access<RW>(h),
						tup.info.t_xmin,
						tup.info.t_xmax,
						tup.info.t_ymin,
						tup.info.t_ymax,
						tup.field.density0.access<RW>(h),
						tup.field.energy0.access<RW>(h),
						tup.field.pressure.access<RW>(h),
						tup.field.viscosity.access<RW>(h),
						tup.field.soundspeed.access<RW>(h),
						tup.field.density1.access<RW>(h),
						tup.field.energy1.access<RW>(h),
						tup.field.xvel0.access<RW>(h),
						tup.field.yvel0.access<RW>(h),
						tup.field.xvel1.access<RW>(h),
						tup.field.yvel1.access<RW>(h),
						tup.field.vol_flux_x.access<RW>(h),
						tup.field.vol_flux_y.access<RW>(h),
						tup.field.mass_flux_x.access<RW>(h),
						tup.field.mass_flux_y.access<RW>(h),
						fields,
						depth);

			}

			if (t_down != external_tile) {
				tile_type &tdown = globals.chunk.tiles[t_down];
				update_tile_halo_b_kernel(
						h,
						tt.info.t_xmin,
						tt.info.t_xmax,
						tt.info.t_ymin,
						tt.info.t_ymax,
						tt.field.density0.access<RW>(h),
						tt.field.energy0.access<RW>(h),
						tt.field.pressure.access<RW>(h),
						tt.field.viscosity.access<RW>(h),
						tt.field.soundspeed.access<RW>(h),
						tt.field.density1.access<RW>(h),
						tt.field.energy1.access<RW>(h),
						tt.field.xvel0.access<RW>(h),
						tt.field.yvel0.access<RW>(h),
						tt.field.xvel1.access<RW>(h),
						tt.field.yvel1.access<RW>(h),
						tt.field.vol_flux_x.access<RW>(h),
						tt.field.vol_flux_y.access<RW>(h),
						tt.field.mass_flux_x.access<RW>(h),
						tt.field.mass_flux_y.access<RW>(h),
						tdown.info.t_xmin,
						tdown.info.t_xmax,
						tdown.info.t_ymin,
						tdown.info.t_ymax,
						tdown.field.density0.access<RW>(h),
						tdown.field.energy0.access<RW>(h),
						tdown.field.pressure.access<RW>(h),
						tdown.field.viscosity.access<RW>(h),
						tdown.field.soundspeed.access<RW>(h),
						tdown.field.density1.access<RW>(h),
						tdown.field.energy1.access<RW>(h),
						tdown.field.xvel0.access<RW>(h),
						tdown.field.yvel0.access<RW>(h),
						tdown.field.xvel1.access<RW>(h),
						tdown.field.yvel1.access<RW>(h),
						tdown.field.vol_flux_x.access<RW>(h),
						tdown.field.vol_flux_y.access<RW>(h),
						tdown.field.mass_flux_x.access<RW>(h),
						tdown.field.mass_flux_y.access<RW>(h),
						fields,
						depth);
			}
		}


		// Update Left Right - Ghost, Real, Ghost - > Real

		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &tt = globals.chunk.tiles[tile];
			int t_left = tt.info.tile_neighbours[tile_left];
			int t_right = tt.info.tile_neighbours[tile_right];

			if (t_left != external_tile) {
				tile_type &tleft = globals.chunk.tiles[t_left];
				update_tile_halo_l_kernel(
						h,
						tt.info.t_xmin,
						tt.info.t_xmax,
						tt.info.t_ymin,
						tt.info.t_ymax,
						tt.field.density0.access<RW>(h),
						tt.field.energy0.access<RW>(h),
						tt.field.pressure.access<RW>(h),
						tt.field.viscosity.access<RW>(h),
						tt.field.soundspeed.access<RW>(h),
						tt.field.density1.access<RW>(h),
						tt.field.energy1.access<RW>(h),
						tt.field.xvel0.access<RW>(h),
						tt.field.yvel0.access<RW>(h),
						tt.field.xvel1.access<RW>(h),
						tt.field.yvel1.access<RW>(h),
						tt.field.vol_flux_x.access<RW>(h),
						tt.field.vol_flux_y.access<RW>(h),
						tt.field.mass_flux_x.access<RW>(h),
						tt.field.mass_flux_y.access<RW>(h),
						tleft.info.t_xmin,
						tleft.info.t_xmax,
						tleft.info.t_ymin,
						tleft.info.t_ymax,
						tleft.field.density0.access<RW>(h),
						tleft.field.energy0.access<RW>(h),
						tleft.field.pressure.access<RW>(h),
						tleft.field.viscosity.access<RW>(h),
						tleft.field.soundspeed.access<RW>(h),
						tleft.field.density1.access<RW>(h),
						tleft.field.energy1.access<RW>(h),
						tleft.field.xvel0.access<RW>(h),
						tleft.field.yvel0.access<RW>(h),
						tleft.field.xvel1.access<RW>(h),
						tleft.field.yvel1.access<RW>(h),
						tleft.field.vol_flux_x.access<RW>(h),
						tleft.field.vol_flux_y.access<RW>(h),
						tleft.field.mass_flux_x.access<RW>(h),
						tleft.field.mass_flux_y.access<RW>(h),
						fields,
						depth);
			}

			if (t_right != external_tile) {
				tile_type &tright = globals.chunk.tiles[t_right];
				update_tile_halo_r_kernel(
						h,
						tt.info.t_xmin,
						tt.info.t_xmax,
						tt.info.t_ymin,
						tt.info.t_ymax,
						tt.field.density0.access<RW>(h),
						tt.field.energy0.access<RW>(h),
						tt.field.pressure.access<RW>(h),
						tt.field.viscosity.access<RW>(h),
						tt.field.soundspeed.access<RW>(h),
						tt.field.density1.access<RW>(h),
						tt.field.energy1.access<RW>(h),
						tt.field.xvel0.access<RW>(h),
						tt.field.yvel0.access<RW>(h),
						tt.field.xvel1.access<RW>(h),
						tt.field.yvel1.access<RW>(h),
						tt.field.vol_flux_x.access<RW>(h),
						tt.field.vol_flux_y.access<RW>(h),
						tt.field.mass_flux_x.access<RW>(h),
						tt.field.mass_flux_y.access<RW>(h),
						tright.info.t_xmin,
						tright.info.t_xmax,
						tright.info.t_ymin,
						tright.info.t_ymax,
						tright.field.density0.access<RW>(h),
						tright.field.energy0.access<RW>(h),
						tright.field.pressure.access<RW>(h),
						tright.field.viscosity.access<RW>(h),
						tright.field.soundspeed.access<RW>(h),
						tright.field.density1.access<RW>(h),
						tright.field.energy1.access<RW>(h),
						tright.field.xvel0.access<RW>(h),
						tright.field.yvel0.access<RW>(h),
						tright.field.xvel1.access<RW>(h),
						tright.field.yvel1.access<RW>(h),
						tright.field.vol_flux_x.access<RW>(h),
						tright.field.vol_flux_y.access<RW>(h),
						tright.field.mass_flux_x.access<RW>(h),
						tright.field.mass_flux_y.access<RW>(h),
						fields,
						depth);
			}
		}

	});
}

