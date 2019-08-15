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


#include "accelerate.h"
#include "timer.h"
#include "sycl_utils.hpp"

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the 
// velocity field.
void accelerate_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		clover::Accessor<double, 2, R>::Type xarea,
		clover::Accessor<double, 2, R>::Type yarea,
		clover::Accessor<double, 2, R>::Type volume,
		clover::Accessor<double, 2, R>::Type density0,
		clover::Accessor<double, 2, R>::Type pressure,
		clover::Accessor<double, 2, R>::Type viscosity,
		clover::Accessor<double, 2, RW>::Type xvel0,
		clover::Accessor<double, 2, RW>::Type yvel0,
		clover::Accessor<double, 2, RW>::Type xvel1,
		clover::Accessor<double, 2, RW>::Type yvel1) {

	double halfdt = 0.5 * dt;

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
//	                                               {x_max + 1 + 2, y_max + 1 + 2});


	clover::par_ranged<class accelerate>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](
			id<2> idx) {

		double stepbymass_s = halfdt / ((density0[offset(idx, -1, -1)] * volume[offset(idx, -1, -1)]
		                                 + density0[offset(idx, -1, 0)] * volume[offset(idx, -1, 0)]
		                                 + density0[idx[0]][idx[1]] * volume[idx[0]][idx[1]]
		                                 + density0[offset(idx, 0, -1)] * volume[offset(idx, 0, -1)])
		                                * 0.25);

		xvel1[idx[0]][idx[1]] = xvel0[idx[0]][idx[1]] - stepbymass_s *
		                          (xarea[idx[0]][idx[1]] * (pressure[idx[0]][idx[1]] - pressure[offset(idx, -1, 0)]) +
		                           xarea[offset(idx, 0, -1)] *
		                           (pressure[offset(idx, 0, -1)] - pressure[offset(idx, -1, -1)]));
		yvel1[idx[0]][idx[1]] = yvel0[idx[0]][idx[1]] - stepbymass_s *
		                          (yarea[idx[0]][idx[1]] * (pressure[idx[0]][idx[1]] - pressure[offset(idx, 0, -1)]) +
		                           yarea[offset(idx, -1, 0)] *
		                           (pressure[offset(idx, -1, 0)] - pressure[offset(idx, -1, -1)]));
		xvel1[idx[0]][idx[1]] = xvel1[idx[0]][idx[1]] - stepbymass_s *
		                          (xarea[idx[0]][idx[1]] * (viscosity[idx[0]][idx[1]] - viscosity[offset(idx, -1, 0)]) +
		                           xarea[offset(idx, 0, -1)] *
		                           (viscosity[offset(idx, 0, -1)] - viscosity[offset(idx, -1,
		                                                                             -1)]));
		yvel1[idx[0]][idx[1]] = yvel1[idx[0]][idx[1]] - stepbymass_s *
		                          (yarea[idx[0]][idx[1]] * (viscosity[idx[0]][idx[1]] - viscosity[offset(idx, 0, -1)]) +
		                           yarea[offset(idx, -1, 0)] *
		                           (viscosity[offset(idx, -1, 0)] - viscosity[offset(idx, -1,
		                                                                             -1)]));


	});
}


//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

	double kernel_time;
	if (globals.profiler_on) kernel_time = timer();


	for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
		tile_type &t = globals.chunk.tiles[tile];


		clover::execute(globals.queue, [&](handler &h) {
			accelerate_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					globals.dt,
					t.field.xarea.access<R>(h),
					t.field.yarea.access<R>(h),
					t.field.volume.access<R>(h),
					t.field.density0.access<R>(h),
					t.field.pressure.access<R>(h),
					t.field.viscosity.access<R>(h),
					t.field.xvel0.access<RW>(h),
					t.field.yvel0.access<RW>(h),
					t.field.xvel1.access<RW>(h),
					t.field.yvel1.access<RW>(h));
		});


	}

	if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;

}
