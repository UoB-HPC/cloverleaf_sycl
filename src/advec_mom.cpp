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


#include "advec_mom.h"

//  @brief Fortran momentum advection kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note that although pre_vol is only set and not used in the update, please
//  leave it in the method.
void advec_mom_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		AccDP2RW::Type vel1,
		AccDP2RW::Type mass_flux_x,
		AccDP2RW::Type vol_flux_x,
		AccDP2RW::Type mass_flux_y,
		AccDP2RW::Type vol_flux_y,
		AccDP2RW::Type volume,
		AccDP2RW::Type density1,
		AccDP2RW::Type node_flux,
		AccDP2RW::Type node_mass_post,
		AccDP2RW::Type node_mass_pre,
		AccDP2RW::Type mom_flux,
		AccDP2RW::Type pre_vol,
		AccDP2RW::Type post_vol,
		AccDP1RW::Type celldx,
		AccDP1RW::Type celldy,
		int which_vel,
		int sweep_number,
		int direction) {

	int mom_sweep = direction + 2 * (sweep_number - 1);

	// DO k=y_min-2,y_max+2
	//   DO j=x_min-2,x_max+2

	Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

	if (mom_sweep == 1) { // x 1
		par_ranged<class APPEND_LN(advec_mom_x1)>(h, policy, [=](id<2> id) {
			post_vol[id] = volume[id] + vol_flux_y[k<1>(id)] - vol_flux_y[id];
			pre_vol[id] = post_vol[id] + vol_flux_x[j<1>(id)] - vol_flux_x[id];
		});
	} else if (mom_sweep == 2) { // y 1
		par_ranged<class APPEND_LN(advec_mom_y1)>(h, policy, [=](id<2> id) {
			post_vol[id] = volume[id] + vol_flux_x[j<1>(id)] - vol_flux_x[id];
			pre_vol[id] = post_vol[id] + vol_flux_y[k<1>(id)] - vol_flux_y[id];
		});
	} else if (mom_sweep == 3) { // x 2
		par_ranged<class APPEND_LN(advec_mom_x1)>(h, policy, [=](id<2> id) {
			post_vol[id] = volume[id];
			pre_vol[id] = post_vol[id] + vol_flux_y[k<1>(id)] - vol_flux_y[id];
		});
	} else if (mom_sweep == 4) { // y 2
		par_ranged<class APPEND_LN(advec_mom_y1)>(h, policy, [=](id<2> id) {
			post_vol[id] = volume[id];
			pre_vol[id] = post_vol[id] + vol_flux_x[j<1>(id)] - vol_flux_x[id];
		});
	}

	if (direction == 1) {
		if (which_vel == 1) {
			// DO k=y_min,y_max+1
			//   DO j=x_min-2,x_max+2
			par_ranged<class advec_mom_dir1_vel1_node_flux>(
					h, {x_min - 2 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2}, [=](id<2> id) {
						// Find staggered mesh mass fluxes, nodal masses and volumes.
						node_flux[id] = 0.25 * (mass_flux_x[k<-1>(id)] + mass_flux_x[id]
						                        + mass_flux_x[jk<1, -1>(id)] +
						                        mass_flux_x[j<1>(id)]);
					});

			// DO k=y_min,y_max+1
			//   DO j=x_min-1,x_max+2
			par_ranged<class advec_mom_dir1_vel1_node_mass_pre>(
					h, {x_min - 1 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2}, [=](id<2> id) {
						// Staggered cell mass post advection
						node_mass_post[id] = 0.25 * (density1[k<-1>(id)] * post_vol[k<-1>(id)]
						                             + density1[id] * post_vol[id]
						                             + density1[jk<-1, -1>(id)] *
						                               post_vol[jk<-1, -1>(id)]
						                             + density1[j<-1>(id)] * post_vol[j<-1>(id)]);
						node_mass_pre[id] =
								node_mass_post[id] - node_flux[j<-1>(id)] + node_flux[id];
					});
		}

		// DO k=y_min,y_max+1
		//  DO j=x_min-1,x_max+1
		par_ranged<class advec_mom_dir1_mom_flux>(
				h, {x_min - 1 + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> id) {

					int upwind, donor, downwind, dif;
					double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

					const int j = id.get(0);
					const int k = id.get(1);

					if (node_flux[id] < 0.0) {
						upwind = j + 2;
						donor = j + 1;
						downwind = j;
						dif = donor;
					} else {
						upwind = j - 1;
						donor = j;
						downwind = j + 1;
						dif = upwind;
					}

					sigma = fabs(node_flux[id]) / (node_mass_pre[donor][k]);
					width = celldx[j];
					vdiffuw = vel1[donor][k] - vel1[upwind][k];
					vdiffdw = vel1[downwind][k] - vel1[donor][k];
					limiter = 0.0;
					if (vdiffuw * vdiffdw > 0.0) {
						auw = fabs(vdiffuw);
						adw = fabs(vdiffdw);
						wind = 1.0;
						if (vdiffdw <= 0.0) wind = -1.0;
						limiter = wind * MIN(MIN(width * ((2.0 - sigma) * adw / width +
						                                  (1.0 + sigma) * auw / celldx[dif]) / 6.0,
						                         auw),
						                     adw);
					}
					advec_vel_s = vel1[donor][k] + (1.0 - sigma) * limiter;
					mom_flux[id] = advec_vel_s * node_flux[id];
				});

		// DO k=y_min,y_max+1
		//   DO j=x_min,x_max+1
		par_ranged<class advec_mom_dir1_vel1>(
				h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> id) {
					vel1[id] = (vel1[id] * node_mass_pre[id] + mom_flux[j<-1>(id)] -
					            mom_flux[id]) /
					           node_mass_post[id];
				});
	} else if (direction == 2) {
		if (which_vel == 1) {
			// DO k=y_min-2,y_max+2
			//   DO j=x_min,x_max+1
			par_ranged<class advec_mom_dir2_vel1_node_flux>(
					h, {x_min + 1, y_min - 2 + 1, x_max + 1 + 2, y_max + 2 + 2}, [=](id<2> id) {
						// Find staggered mesh mass fluxes and nodal masses and volumes.
						node_flux[id] = 0.25 * (mass_flux_y[j<-1>(id)] + mass_flux_y[id]
						                        + mass_flux_y[jk<-1, 1>(id)] +
						                        mass_flux_y[k<1>(id)]);
					});


			// DO k=y_min-1,y_max+2
			//   DO j=x_min,x_max+1
			par_ranged<class advec_mom_dir2_vel1_node_mass_pre>(
					h, {x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 2 + 2}, [=](id<2> id) {
						node_mass_post[id] = 0.25 * (density1[k<-1>(id)] * post_vol[k<-1>(id)]
						                             + density1[id] * post_vol[id]
						                             + density1[jk<-1, -1>(id)] *
						                               post_vol[jk<-1, -1>(id)]
						                             + density1[j<-1>(id)] * post_vol[j<-1>(id)]);
						node_mass_pre[id] =
								node_mass_post[id] - node_flux[k<-1>(id)] + node_flux[id];
					});
		}

		// DO k=y_min-1,y_max+1
		//   DO j=x_min,x_max+1
		par_ranged<class advec_mom_dir2_mom_flux>(
				h, {x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> id) {

					int upwind, donor, downwind, dif;
					double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

					const int j = id.get(0);
					const int k = id.get(1);

					if (node_flux[id] < 0.0) {
						upwind = k + 2;
						donor = k + 1;
						downwind = k;
						dif = donor;
					} else {
						upwind = k - 1;
						donor = k;
						downwind = k + 1;
						dif = upwind;
					}


					sigma = fabs(node_flux[id]) / (node_mass_pre[j][donor]);
					width = celldy[k];
					vdiffuw = vel1[j][donor] - vel1[j][upwind];
					vdiffdw = vel1[j][downwind] - vel1[j][donor];
					limiter = 0.0;
					if (vdiffuw * vdiffdw > 0.0) {
						auw = fabs(vdiffuw);
						adw = fabs(vdiffdw);
						wind = 1.0;
						if (vdiffdw <= 0.0) wind = -1.0;
						limiter = wind * MIN(MIN(width * ((2.0 - sigma) * adw / width +
						                                  (1.0 + sigma) * auw / celldy[dif]) / 6.0,
						                         auw),
						                     adw);
					}
					advec_vel_s = vel1[j][donor] + (1.0 - sigma) * limiter;
					mom_flux[id] = advec_vel_s * node_flux[id];
				});


		// DO k=y_min,y_max+1
		//   DO j=x_min,x_max+1
		par_ranged<class advec_mom_dir2_vel1>(
				h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> id) {
					vel1[id] = (vel1[id] * node_mass_pre[id] + mom_flux[k<-1>(id)] - mom_flux[id]) /
					           node_mass_post[id];
				});
	}
}


//  @brief Momentum advection driver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum advection kernel.
void advec_mom_driver(global_variables &globals, int tile, int which_vel, int direction,
                      int sweep_number) {

	execute(globals.queue, [&](handler &h) {
		tile_type &t = globals.chunk.tiles[tile];
		if (which_vel == 1) {
			advec_mom_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					t.field.xvel1.access<RW>(h),
					t.field.mass_flux_x.access<RW>(h),
					t.field.vol_flux_x.access<RW>(h),
					t.field.mass_flux_y.access<RW>(h),
					t.field.vol_flux_y.access<RW>(h),
					t.field.volume.access<RW>(h),
					t.field.density1.access<RW>(h),
					t.field.work_array1.access<RW>(h),
					t.field.work_array2.access<RW>(h),
					t.field.work_array3.access<RW>(h),
					t.field.work_array4.access<RW>(h),
					t.field.work_array5.access<RW>(h),
					t.field.work_array6.access<RW>(h),
					t.field.celldx.access<RW>(h),
					t.field.celldy.access<RW>(h),
					which_vel,
					sweep_number,
					direction);
		} else {
			advec_mom_kernel(
					h,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					t.field.yvel1.access<RW>(h),
					t.field.mass_flux_x.access<RW>(h),
					t.field.vol_flux_x.access<RW>(h),
					t.field.mass_flux_y.access<RW>(h),
					t.field.vol_flux_y.access<RW>(h),
					t.field.volume.access<RW>(h),
					t.field.density1.access<RW>(h),
					t.field.work_array1.access<RW>(h),
					t.field.work_array2.access<RW>(h),
					t.field.work_array3.access<RW>(h),
					t.field.work_array4.access<RW>(h),
					t.field.work_array5.access<RW>(h),
					t.field.work_array6.access<RW>(h),
					t.field.celldx.access<RW>(h),
					t.field.celldy.access<RW>(h),
					which_vel,
					sweep_number,
					direction);
		}
	});

}


