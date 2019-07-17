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


#include "advec_cell.h"
#include "sycl_utils.hpp"


//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(
		handler &h,
		int x_min,
		int x_max,
		int y_min,
		int y_max,
		int dir,
		int sweep_number,
		Accessor<double, 1, RW>::Type vertexdx,
		Accessor<double, 1, RW>::Type vertexdy,
		Accessor<double, 2, RW>::Type volume,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type mass_flux_x,
		Accessor<double, 2, RW>::Type vol_flux_x,
		Accessor<double, 2, RW>::Type mass_flux_y,
		Accessor<double, 2, RW>::Type vol_flux_y,
		Accessor<double, 2, RW>::Type pre_vol,
		Accessor<double, 2, RW>::Type post_vol,
		Accessor<double, 2, RW>::Type pre_mass,
		Accessor<double, 2, RW>::Type post_mass,
		Accessor<double, 2, RW>::Type advec_vol,
		Accessor<double, 2, RW>::Type post_ener,
		Accessor<double, 2, RW>::Type ener_flux) {

	const double one_by_six = 1.0 / 6.0;

	if (dir == g_xdir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2

		const Range2d policy(x_min - 2 + 1, y_min - 2 + 1,
		                     x_max + 2 + 2, y_max + 2 + 2);

		if (sweep_number == 1) {
			par_ranged<class advec_cell_xdir_seq1>(h, policy, [=](id<2> id) {

				pre_vol[id] = volume[id] +
				              (vol_flux_x[j<1>(id)] - vol_flux_x[id] + vol_flux_y[k<1>(id)] -
				               vol_flux_y[id]);
				post_vol[id] = pre_vol[id] - (vol_flux_x[j<1>(id)] - vol_flux_x[id]);
			});

		} else {
			par_ranged<class advec_cell_xdir_sne1>(h, policy, [=](id<2> id) {
				pre_vol[id] = volume[id] + vol_flux_x[j<1>(id)] - vol_flux_x[id];
				post_vol[id] = volume[id];
			});
		}

		// DO k=y_min,y_max
		//   DO j=x_min,x_max+2
		par_ranged<class advec_cell_xdir_ener_flux>(
				h, {x_min + 1, y_min + 1, x_max + 2 + 2, y_max + 2}, [=](id<2> id) {


					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;

					const int j = id.get(0);
					const int k = id.get(1);

					if (vol_flux_x[id] > 0.0) {
						upwind = j - 2;
						donor = j - 1;
						downwind = j;
						dif = donor;
					} else {
						upwind = MIN(j + 1, x_max + 2);
						donor = j;
						downwind = j - 1;
						dif = upwind;
					}


					sigmat = fabs(vol_flux_x[id]) / pre_vol[donor][k];
					sigma3 = (1.0 + sigmat) * (vertexdx[j] / vertexdx[dif]);
					sigma4 = 2.0 - sigmat;

					sigma = sigmat;
					sigmav = sigmat;

					diffuw = density1[donor][k] - density1[upwind][k];
					diffdw = density1[downwind][k] - density1[donor][k];
					wind = 1.0;
					if (diffdw <= 0.0) wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind *
						          MIN(MIN(fabs(diffuw), fabs(diffdw)), one_by_six *
								          (sigma3 *
								           fabs(diffuw) +
								           sigma4 *
								           fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					mass_flux_x[id] = vol_flux_x[id] * (density1[donor][k] + limiter);

					sigmam = fabs(mass_flux_x[id]) / (density1[donor][k] * pre_vol[donor][k]);
					diffuw = energy1[donor][k] - energy1[upwind][k];
					diffdw = energy1[downwind][k] - energy1[donor][k];
					wind = 1.0;
					if (diffdw <= 0.0) wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) * wind *
						          MIN(MIN(fabs(diffuw), fabs(diffdw)), one_by_six *
								          (sigma3 *
								           fabs(diffuw) +
								           sigma4 *
								           fabs(diffdw)));
					} else {
						limiter = 0.0;
					}

					ener_flux[id] = mass_flux_x[id] * (energy1[donor][k] + limiter);

				});

		// DO k=y_min,y_max
		//   DO j=x_min,x_max
		par_ranged<class advec_cell_xdir_d1e1>(
				h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> id) {
					double pre_mass_s = density1[id] * pre_vol[id];
					double post_mass_s = pre_mass_s + mass_flux_x[id] - mass_flux_x[j<1>(id)];
					double post_ener_s =
							(energy1[id] * pre_mass_s + ener_flux[id] - ener_flux[j<1>(id)]) /
							post_mass_s;
					double advec_vol_s = pre_vol[id] + vol_flux_x[id] - vol_flux_x[j<1>(id)];
					density1[id] = post_mass_s / advec_vol_s;
					energy1[id] = post_ener_s;
				});
	} else if (dir == g_ydir) {

		// DO k=y_min-2,y_max+2
		//   DO j=x_min-2,x_max+2
		Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

		if (sweep_number == 1) {
			par_ranged<class APPEND_LN(advec_cell_ydir_s1)>(h, policy, [=](id<2> id) {
				pre_vol[id] = volume[id] +
				              (vol_flux_y[k<1>(id)] - vol_flux_y[id] + vol_flux_x[j<1>(id)] -
				               vol_flux_x[id]);
				post_vol[id] = pre_vol[id] - (vol_flux_y[k<1>(id)] - vol_flux_y[id]);
			});
		} else {
			par_ranged<class APPEND_LN(advec_cell_ydir_s1)>(h, policy, [=](id<2> id) {
				pre_vol[id] = volume[id] + vol_flux_y[k<1>(id)] - vol_flux_y[id];
				post_vol[id] = volume[id];
			});
		}

		// DO k=y_min,y_max+2
		//   DO j=x_min,x_max

		par_ranged<class advec_cell_ydir_ener_flux>(
				h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2 + 2}, [=](id<2> id) {
					int upwind, donor, downwind, dif;
					double sigmat, sigma3, sigma4, sigmav, sigma, sigmam, diffuw, diffdw, limiter, wind;

					const int j = id.get(0);
					const int k = id.get(1);

					if (vol_flux_y[id] > 0.0) {
						upwind = k - 2;
						donor = k - 1;
						downwind = k;
						dif = donor;
					} else {
						upwind = MIN(k + 1, y_max + 2);
						donor = k;
						downwind = k - 1;
						dif = upwind;
					}

					sigmat = fabs(vol_flux_y[id]) / pre_vol[j][donor];
					sigma3 = (1.0 + sigmat) * (vertexdy[k] / vertexdy[dif]);
					sigma4 = 2.0 - sigmat;

					sigma = sigmat;
					sigmav = sigmat;

					diffuw = density1[j][donor] - density1[j][upwind];
					diffdw = density1[j][downwind] - density1[j][donor];
					wind = 1.0;
					if (diffdw <= 0.0) wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmav) * wind * MIN(MIN(fabs(diffuw), fabs(diffdw)),
						                                      one_by_six * (sigma3 * fabs(diffuw) +
						                                                    sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					mass_flux_y[id] = vol_flux_y[id] * (density1[j][donor] + limiter);

					sigmam = fabs(mass_flux_y[id]) / (density1[j][donor] * pre_vol[j][donor]);
					diffuw = energy1[j][donor] - energy1[j][upwind];
					diffdw = energy1[j][downwind] - energy1[j][donor];
					wind = 1.0;
					if (diffdw <= 0.0) wind = -1.0;
					if (diffuw * diffdw > 0.0) {
						limiter = (1.0 - sigmam) * wind * MIN(MIN(fabs(diffuw), fabs(diffdw)),
						                                      one_by_six * (sigma3 * fabs(diffuw) +
						                                                    sigma4 * fabs(diffdw)));
					} else {
						limiter = 0.0;
					}
					ener_flux[id] = mass_flux_y[id] * (energy1[j][donor] + limiter);
				});

		// DO k=y_min,y_max
		//   DO j=x_min,x_max
		par_ranged<class advec_cell_ydir_e1d1>(
				h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> id) {

					double pre_mass_s = density1[id] * pre_vol[id];
					double post_mass_s = pre_mass_s + mass_flux_y[id] - mass_flux_y[k<1>(id)];
					double post_ener_s =
							(energy1[id] * pre_mass_s + ener_flux[id] - ener_flux[k<1>(id)]) /
							post_mass_s;
					double advec_vol_s = pre_vol[id] + vol_flux_y[id] - vol_flux_y[k<1>(id)];
					density1[id] = post_mass_s / advec_vol_s;
					energy1[id] = post_ener_s;
				});
	}

}


//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

	tile_type &t = globals.chunk.tiles[tile];
	execute(globals.queue, [&](handler &h) {
		advec_cell_kernel(
				h,
				t.info.t_xmin,
				t.info.t_xmax,
				t.info.t_ymin,
				t.info.t_ymax,
				direction,
				sweep_number,
				t.field.vertexdx.access<RW>(h),
				t.field.vertexdy.access<RW>(h),
				t.field.volume.access<RW>(h),
				t.field.density1.access<RW>(h),
				t.field.energy1.access<RW>(h),
				t.field.mass_flux_x.access<RW>(h),
				t.field.vol_flux_x.access<RW>(h),
				t.field.mass_flux_y.access<RW>(h),
				t.field.vol_flux_y.access<RW>(h),
				t.field.work_array1.access<RW>(h),
				t.field.work_array2.access<RW>(h),
				t.field.work_array3.access<RW>(h),
				t.field.work_array4.access<RW>(h),
				t.field.work_array5.access<RW>(h),
				t.field.work_array6.access<RW>(h),
				t.field.work_array7.access<RW>(h));

	});


}

