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


#include "ideal_gas.h"

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		AccDP2RW::Type density,
		AccDP2RW::Type energy,
		AccDP2RW::Type pressure,
		AccDP2RW::Type soundspeed) {

	// DO k=y_min,y_max
	//   DO j=x_min,x_max

//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

	par_ranged<class ideal_gas>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> id) {
		double v = 1.0 / density[id];
		pressure[id] = (1.4 - 1.0) * density[id] * energy[id];
		double pressurebyenergy = (1.4 - 1.0) * density[id];
		double pressurebyvolume = -density[id] * pressure[id];
		double sound_speed_squared = v * v * (pressure[id] * pressurebyenergy - pressurebyvolume);
		soundspeed[id] = sqrt(sound_speed_squared);
	});

}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables &globals, const int tile, bool predict) {
	if (DEBUG) std::cout << "ideal_gas(tile " << tile << ")" << std::endl;

	tile_type &t = globals.chunk.tiles[tile];

	execute(globals.queue, [&](handler &h) {

		if (!predict) {
			ideal_gas_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					t.field.density0.access<RW>(h),
					t.field.energy0.access<RW>(h),
					t.field.pressure.access<RW>(h),
					t.field.soundspeed.access<RW>(h)
			);
		} else {
			ideal_gas_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					t.field.density1.access<RW>(h),
					t.field.energy1.access<RW>(h),
					t.field.pressure.access<RW>(h),
					t.field.soundspeed.access<RW>(h)
			);
		}
	});

}

