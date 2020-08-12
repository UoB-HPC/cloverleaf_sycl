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


#include "PdV.h"
#include "timer.h"
#include "comms.h"
#include "report.h"
#include "ideal_gas.h"
#include "update_halo.h"
#include "revert.h"
#include "sycl_utils.hpp"

//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in energy and density in a cell using the
//  change on cell volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void PdV_kernel(
		handler &h,
		bool predict,
		int x_min, int x_max, int y_min, int y_max,
		double dt,
		clover::Accessor<double, 2, RW>::Type xarea,
		clover::Accessor<double, 2, RW>::Type yarea,
		clover::Accessor<double, 2, RW>::Type volume,
		clover::Accessor<double, 2, RW>::Type density0,
		clover::Accessor<double, 2, RW>::Type density1,
		clover::Accessor<double, 2, RW>::Type energy0,
		clover::Accessor<double, 2, RW>::Type energy1,
		clover::Accessor<double, 2, RW>::Type pressure,
		clover::Accessor<double, 2, RW>::Type viscosity,
		clover::Accessor<double, 2, RW>::Type xvel0,
		clover::Accessor<double, 2, RW>::Type xvel1,
		clover::Accessor<double, 2, RW>::Type yvel0,
		clover::Accessor<double, 2, RW>::Type yvel1,
		clover::Accessor<double, 2, RW>::Type volume_change) {


	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	clover::Range2d policy(x_min + 1, y_min + 1, x_max + 2, y_max + 2);

	if (predict) {

		clover::par_ranged<class PdV_predict_true>(h, policy, [=](id<2> idx) {


			double left_flux = (xarea[idx] * (xvel0[idx] + xvel0[clover::offset(idx, 0, 1)]
			                                  + xvel0[idx] + xvel0[clover::offset(idx, 0, 1)])) * 0.25 * dt * 0.5;

			double right_flux = (xarea[clover::offset(idx, 1, 0)] * (xvel0[clover::offset(idx, 1, 0)] + xvel0[clover::offset(idx, 1, 1)]
			                                                 + xvel0[clover::offset(idx, 1, 0)] + xvel0[clover::offset(idx, 1, 1)])) *
			                    0.25 * dt * 0.5;

			double bottom_flux = (yarea[idx] * (yvel0[idx] + yvel0[clover::offset(idx, 1, 0)]
			                                    + yvel0[idx] + yvel0[clover::offset(idx, 1, 0)])) * 0.25 * dt *
			                     0.5;

			double top_flux = (yarea[clover::offset(idx, 0, 1)] * (yvel0[clover::offset(idx, 0, 1)] + yvel0[clover::offset(idx, 1, 1)]
			                                               + yvel0[clover::offset(idx, 0, 1)] + yvel0[clover::offset(idx, 1, 1)])) *
			                  0.25 *
			                  dt * 0.5;

			double total_flux = right_flux - left_flux + top_flux - bottom_flux;

			double volume_change_s = volume[idx] / (volume[idx] + total_flux);

			double min_cell_volume =
					sycl::fmin(sycl::fmin(volume[idx] + right_flux - left_flux + top_flux - bottom_flux,
					                      volume[idx] + right_flux - left_flux), volume[idx] + top_flux - bottom_flux);

			double recip_volume = 1.0 / volume[idx];

			double energy_change =
					(pressure[idx] / density0[idx] + viscosity[idx] / density0[idx]) *
					total_flux * recip_volume;

			energy1[idx] = energy0[idx] - energy_change;

			density1[idx] = density0[idx] * volume_change_s;

		});

	} else {

		clover::par_ranged<class PdV_predict_false>(h, policy, [=](id<2> idx) {

			double left_flux = (xarea[idx] * (xvel0[idx] + xvel0[clover::offset(idx, 0, 1)]
			                                  + xvel1[idx] + xvel1[clover::offset(idx, 0, 1)])) * 0.25 * dt;

			double right_flux = (xarea[clover::offset(idx, 1, 0)] * (xvel0[clover::offset(idx, 1, 0)] + xvel0[clover::offset(idx, 1, 1)]
			                                                 + xvel1[clover::offset(idx, 1, 0)] + xvel1[clover::offset(idx, 1, 1)])) *
			                    0.25 * dt;

			double bottom_flux = (yarea[idx] * (yvel0[idx] + yvel0[clover::offset(idx, 1, 0)]
			                                    + yvel1[idx] + yvel1[clover::offset(idx, 1, 0)])) * 0.25 * dt;

			double top_flux = (yarea[clover::offset(idx, 0, 1)] * (yvel0[clover::offset(idx, 0, 1)] + yvel0[clover::offset(idx, 1, 1)]
			                                               + yvel1[clover::offset(idx, 0, 1)] + yvel1[clover::offset(idx, 1, 1)])) *
			                  0.25 *
			                  dt;

			double total_flux = right_flux - left_flux + top_flux - bottom_flux;

			double volume_change_s = volume[idx] / (volume[idx] + total_flux);

			double min_cell_volume =
					sycl::fmin(sycl::fmin(volume[idx] + right_flux - left_flux + top_flux - bottom_flux,
					                      volume[idx] + right_flux - left_flux), volume[idx] + top_flux - bottom_flux);

			double recip_volume = 1.0 / volume[idx];

			double energy_change =
					(pressure[idx] / density0[idx] + viscosity[idx] / density0[idx]) *
					total_flux * recip_volume;

			energy1[idx] = energy0[idx] - energy_change;

			density1[idx] = density0[idx] * volume_change_s;

		});
	}

}


//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(global_variables &globals, bool predict) {

	double kernel_time = 0;
	if (globals.profiler_on) kernel_time = timer();

	globals.error_condition = 0;

	clover::execute(globals.queue, [&](handler &h) {
		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &t = globals.chunk.tiles[tile];
			PdV_kernel(
					h,
					predict,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					globals.dt,
					t.field.xarea.access<RW>(h),
					t.field.yarea.access<RW>(h),
					t.field.volume.access<RW>(h),
					t.field.density0.access<RW>(h),
					t.field.density1.access<RW>(h),
					t.field.energy0.access<RW>(h),
					t.field.energy1.access<RW>(h),
					t.field.pressure.access<RW>(h),
					t.field.viscosity.access<RW>(h),
					t.field.xvel0.access<RW>(h),
					t.field.xvel1.access<RW>(h),
					t.field.yvel0.access<RW>(h),
					t.field.yvel1.access<RW>(h),
					t.field.work_array1.access<RW>(h));

		}
	});


	clover_check_error(globals.error_condition);
	if (globals.profiler_on) globals.profiler.PdV += timer() - kernel_time;

	if (globals.error_condition == 1) {
		report_error((char *) "PdV", (char *) "error in PdV");
	}

	if (predict) {
		if (globals.profiler_on) kernel_time = timer();
		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			ideal_gas(globals, tile, true);
		}

		if (globals.profiler_on) globals.profiler.ideal_gas += timer() - kernel_time;

		int fields[NUM_FIELDS];
		for (int & field : fields) field = 0;
		fields[field_pressure] = 1;
		update_halo(globals, fields, 1);
	}

	if (predict) {
		if (globals.profiler_on) kernel_time = timer();
		revert(globals);
		if (globals.profiler_on) globals.profiler.revert += timer() - kernel_time;
	}

}


