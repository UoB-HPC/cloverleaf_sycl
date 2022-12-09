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
#include "sycl_utils.hpp"

//  @brief Fortran momentum advection kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note that although pre_vol is only set and not used in the update, please
//  leave it in the method.
void advec_mom_kernel(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max, clover::Buffer<double, 2> vel1,
                      clover::Buffer<double, 2> mass_flux_x, clover::Buffer<double, 2> vol_flux_x,
                      clover::Buffer<double, 2> mass_flux_y, clover::Buffer<double, 2> vol_flux_y,
                      clover::Buffer<double, 2> volume, clover::Buffer<double, 2> density1,
                      clover::Buffer<double, 2> node_flux, clover::Buffer<double, 2> node_mass_post,
                      clover::Buffer<double, 2> node_mass_pre, clover::Buffer<double, 2> mom_flux,
                      clover::Buffer<double, 2> pre_vol, clover::Buffer<double, 2> post_vol,
                      clover::Buffer<double, 1> celldx, clover::Buffer<double, 1> celldy, int which_vel,
                      int sweep_number, int direction) {

  int mom_sweep = direction + 2 * (sweep_number - 1);

  // DO k=y_min-2,y_max+2
  //   DO j=x_min-2,x_max+2

  clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

  if (mom_sweep == 1) { // x 1

    clover::par_ranged2(queue, policy, [=](const int i, const int j) {
      post_vol(i, j) = volume(i, j) + vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j);
      pre_vol(i, j) = post_vol(i, j) + vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j);
    });
  } else if (mom_sweep == 2) { // y 1

    clover::par_ranged2(queue, policy, [=](const int i, const int j) {
      post_vol(i, j) = volume(i, j) + vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j);
      pre_vol(i, j) = post_vol(i, j) + vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j);
    });
  } else if (mom_sweep == 3) { // x 2

    clover::par_ranged2(queue, policy, [=](const int i, const int j) {
      post_vol(i, j) = volume(i, j);
      pre_vol(i, j) = post_vol(i, j) + vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j);
    });
  } else if (mom_sweep == 4) { // y 2

    clover::par_ranged2(queue, policy, [=](const int i, const int j) {
      post_vol(i, j) = volume(i, j);
      pre_vol(i, j) = post_vol(i, j) + vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j);
    });
  }

  if (direction == 1) {
    if (which_vel == 1) {
      // DO k=y_min,y_max+1
      //   DO j=x_min-2,x_max+2

      clover::par_ranged2(queue, Range2d{x_min - 2 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2},
                          [=](const int i, const int j) {
                            // Find staggered mesh mass fluxes, nodal masses and volumes.
                            node_flux(i, j) = 0.25 * (mass_flux_x(i + 0, j - 1) + mass_flux_x(i, j) +
                                                      mass_flux_x(i + 1, j - 1) + mass_flux_x(i + 1, j + 0));
                          });

      // DO k=y_min,y_max+1
      //   DO j=x_min-1,x_max+2

      clover::par_ranged2(
          queue, Range2d{x_min - 1 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2}, [=](const int i, const int j) {
            // Staggered cell mass post advection
            node_mass_post(i, j) =
                0.25 *
                (density1(i + 0, j - 1) * post_vol(i + 0, j - 1) + density1(i, j) * post_vol(i, j) +
                 density1(i - 1, j - 1) * post_vol(i - 1, j - 1) + density1(i - 1, j + 0) * post_vol(i - 1, j + 0));
            node_mass_pre(i, j) = node_mass_post(i, j) - node_flux(i - 1, j + 0) + node_flux(i, j);
          });
    }

    // DO k=y_min,y_max+1
    //  DO j=x_min-1,x_max+1

    clover::par_ranged2(
        queue, Range2d{x_min - 1 + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](const int x, const int y) {
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

          const int j = x;
          const int k = y;

          if (node_flux(x, y) < 0.0) {
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

          sigma = sycl::fabs(node_flux(x, y)) / (node_mass_pre(donor, k));
          width = celldx[j];
          vdiffuw = vel1(donor, k) - vel1(upwind, k);
          vdiffdw = vel1(downwind, k) - vel1(donor, k);
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = sycl::fabs(vdiffuw);
            adw = sycl::fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter =
                wind *
                sycl::fmin(
                    sycl::fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx[dif]) / 6.0, auw),
                    adw);
          }
          advec_vel_s = vel1(donor, k) + (1.0 - sigma) * limiter;
          mom_flux(x, y) = advec_vel_s * node_flux(x, y);
        });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    clover::par_ranged2(queue, Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2},
                        [=](const int i, const int j) {
                          vel1(i, j) = (vel1(i, j) * node_mass_pre(i, j) + mom_flux(i - 1, j + 0) - mom_flux(i, j)) /
                                       node_mass_post(i, j);
                        });
  } else if (direction == 2) {
    if (which_vel == 1) {
      // DO k=y_min-2,y_max+2
      //   DO j=x_min,x_max+1

      clover::par_ranged2(queue, Range2d{x_min + 1, y_min - 2 + 1, x_max + 1 + 2, y_max + 2 + 2},
                          [=](const int i, const int j) {
                            // Find staggered mesh mass fluxes and nodal masses and volumes.
                            node_flux(i, j) = 0.25 * (mass_flux_y(i - 1, j + 0) + mass_flux_y(i, j) +
                                                      mass_flux_y(i - 1, j + 1) + mass_flux_y(i + 0, j + 1));
                          });

      // DO k=y_min-1,y_max+2
      //   DO j=x_min,x_max+1

      clover::par_ranged2(
          queue, Range2d{x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 2 + 2}, [=](const int i, const int j) {
            node_mass_post(i, j) =
                0.25 *
                (density1(i + 0, j - 1) * post_vol(i + 0, j - 1) + density1(i, j) * post_vol(i, j) +
                 density1(i - 1, j - 1) * post_vol(i - 1, j - 1) + density1(i - 1, j + 0) * post_vol(i - 1, j + 0));
            node_mass_pre(i, j) = node_mass_post(i, j) - node_flux(i + 0, j - 1) + node_flux(i, j);
          });
    }

    // DO k=y_min-1,y_max+1
    //   DO j=x_min,x_max+1

    clover::par_ranged2(
        queue, Range2d{x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](const int x, const int y) {
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

          const int j = x;
          const int k = y;

          if (node_flux(x, y) < 0.0) {
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

          sigma = sycl::fabs(node_flux(x, y)) / (node_mass_pre(j, donor));
          width = celldy[k];
          vdiffuw = vel1(j, donor) - vel1(j, upwind);
          vdiffdw = vel1(j, downwind) - vel1(j, donor);
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = sycl::fabs(vdiffuw);
            adw = sycl::fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter =
                wind *
                sycl::fmin(
                    sycl::fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy[dif]) / 6.0, auw),
                    adw);
          }
          advec_vel_s = vel1(j, donor) + (1.0 - sigma) * limiter;
          mom_flux(x, y) = advec_vel_s * node_flux(x, y);
        });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    clover::par_ranged2(queue, Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2},
                        [=](const int i, const int j) {
                          vel1(i, j) = (vel1(i, j) * node_mass_pre(i, j) + mom_flux(i + 0, j - 1) - mom_flux(i, j)) /
                                       node_mass_post(i, j);
                        });
  }
}

//  @brief Momentum advection driver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum advection kernel.
void advec_mom_driver(global_variables &globals, int tile, int which_vel, int direction, int sweep_number) {

  tile_type &t = globals.chunk.tiles[tile];
  if (which_vel == 1) {
    advec_mom_kernel(globals.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1,
                     t.field.mass_flux_x, t.field.vol_flux_x, t.field.mass_flux_y, t.field.vol_flux_y, t.field.volume,
                     t.field.density1, t.field.work_array1, t.field.work_array2, t.field.work_array3,
                     t.field.work_array4, t.field.work_array5, t.field.work_array6, t.field.celldx, t.field.celldy,
                     which_vel, sweep_number, direction);
  } else {
    advec_mom_kernel(globals.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1,
                     t.field.mass_flux_x, t.field.vol_flux_x, t.field.mass_flux_y, t.field.vol_flux_y, t.field.volume,
                     t.field.density1, t.field.work_array1, t.field.work_array2, t.field.work_array3,
                     t.field.work_array4, t.field.work_array5, t.field.work_array6, t.field.celldx, t.field.celldy,
                     which_vel, sweep_number, direction);
  }
}
