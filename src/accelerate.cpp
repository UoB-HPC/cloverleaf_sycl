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

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the 
// velocity field.
void accelerate_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		AccDP2RW::Type xarea,
		AccDP2RW::Type yarea,
		AccDP2RW::Type volume,
		AccDP2RW::Type density0,
		AccDP2RW::Type pressure,
		AccDP2RW::Type viscosity,
		AccDP2RW::Type xvel0,
		AccDP2RW::Type yvel0,
		AccDP2RW::Type xvel1,
		AccDP2RW::Type yvel1) {

	double halfdt = 0.5 * dt;

	// DO k=y_min,y_max+1
	//   DO j=x_min,x_max+1
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
//	                                               {x_max + 1 + 2, y_max + 1 + 2});


	par_ranged<class accelerate>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](
			id<2> id) {

		double stepbymass_s = halfdt / ((density0[jk<-1, -1>(id)] * volume[jk<-1, -1>(id)]
		                                 + density0[j<-1>(id)] * volume[j<-1>(id)]
		                                 + density0[id] * volume[id]
		                                 + density0[k<-1>(id)] * volume[k<-1>(id)])
		                                * 0.25);

		xvel1[id] =
				xvel0[id] - stepbymass_s * (xarea[id] * (pressure[id] - pressure[k<-1>(id)])
				                            + xarea[j<-1>(id)] *
				                              (pressure[j<-1>(id)] -
				                               pressure[jk<-1, -1>(id)]));
		yvel1[id] =
				yvel0[id] - stepbymass_s * (yarea[id] * (pressure[id] - pressure[j<-1>(id)])
				                            + yarea[k<-1>(id)] *
				                              (pressure[k<-1>(id)] -
				                               pressure[jk<-1, -1>(id)]));
		xvel1[id] =
				xvel1[id] -
				stepbymass_s * (xarea[id] * (viscosity[id] - viscosity[k<-1>(id)])
				                + xarea[j<-1>(id)] *
				                  (viscosity[j<-1>(id)] - viscosity[jk<-1, -1>(id)]));
		yvel1[id] =
				yvel1[id] -
				stepbymass_s * (yarea[id] * (viscosity[id] - viscosity[j<-1>(id)])
				                + yarea[k<-1>(id)] *
				                  (viscosity[k<-1>(id)] - viscosity[jk<-1, -1>(id)]));


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


		execute(globals.queue, [&](handler &h) {
			accelerate_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					globals.dt,
					t.field.xarea.access<RW>(h),
					t.field.yarea.access<RW>(h),
					t.field.volume.access<RW>(h),
					t.field.density0.access<RW>(h),
					t.field.pressure.access<RW>(h),
					t.field.viscosity.access<RW>(h),
					t.field.xvel0.access<RW>(h),
					t.field.yvel0.access<RW>(h),
					t.field.xvel1.access<RW>(h),
					t.field.yvel1.access<RW>(h));
		});


	}

	if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;

}
