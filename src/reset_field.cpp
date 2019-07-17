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


#include "reset_field.h"
#include "timer.h"
#include "sycl_utils.hpp"

//  @brief Fortran reset field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of step filed data to the begining of
//  step data, ready for the next timestep.
void reset_field_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		Accessor<double, 2, RW>::Type density0,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy0,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type xvel0,
		Accessor<double, 2, RW>::Type xvel1,
		Accessor<double, 2, RW>::Type yvel0,
		Accessor<double, 2, RW>::Type yvel1) {

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	par_ranged<class reset_field_1>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](
			id<2> idx) {
		density0[idx] = density1[idx];
		energy0[idx] = energy1[idx];

	});

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
	par_ranged<class reset_field_2>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](
			id<2> idx) {
		xvel0[idx] = xvel1[idx];
		yvel0[idx] = yvel1[idx];
	});

}


//  @brief Reset field driver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset kernel.
void reset_field(global_variables &globals) {

	double kernel_time;
	if (globals.profiler_on) kernel_time = timer();


	execute(globals.queue, [&](handler &h) {

		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

			tile_type &t = globals.chunk.tiles[tile];
			reset_field_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					t.field.density0.access<RW>(h),
					t.field.density1.access<RW>(h),
					t.field.energy0.access<RW>(h),
					t.field.energy1.access<RW>(h),
					t.field.xvel0.access<RW>(h),
					t.field.xvel1.access<RW>(h),
					t.field.yvel0.access<RW>(h),
					t.field.yvel1.access<RW>(h));
		}
	});


	if (globals.profiler_on) globals.profiler.reset += timer() - kernel_time;
}

