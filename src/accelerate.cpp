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
#include "sycl_utils.hpp"
#include "timer.h"

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the
// velocity field.
void accelerate_kernel(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max, double dt,
                       clover::Buffer<double, 2> xarea, clover::Buffer<double, 2> yarea,
                       clover::Buffer<double, 2> volume, clover::Buffer<double, 2> density0,
                       clover::Buffer<double, 2> pressure, clover::Buffer<double, 2> viscosity,
                       clover::Buffer<double, 2> xvel0, clover::Buffer<double, 2> yvel0,
                       clover::Buffer<double, 2> xvel1, clover::Buffer<double, 2> yvel1) {

  double halfdt = 0.5 * dt;

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
  //	                                               {x_max + 1 + 2, y_max + 1 + 2});

  clover::par_ranged2(
      queue, Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](const int i, const int j) {
        double stepbymass_s =
            halfdt / ((density0(i - 1, j - 1) * volume(i - 1, j - 1) + density0(i - 1, j + 0) * volume(i - 1, j + 0) +
                       density0(i, j) * volume(i, j) + density0(i + 0, j - 1) * volume(i + 0, j - 1)) *
                      0.25);

        xvel1(i, j) =
            xvel0(i, j) - stepbymass_s * (xarea(i, j) * (pressure(i, j) - pressure(i - 1, j + 0)) +
                                          xarea(i + 0, j - 1) * (pressure(i + 0, j - 1) - pressure(i - 1, j - 1)));
        yvel1(i, j) =
            yvel0(i, j) - stepbymass_s * (yarea(i, j) * (pressure(i, j) - pressure(i + 0, j - 1)) +
                                          yarea(i - 1, j + 0) * (pressure(i - 1, j + 0) - pressure(i - 1, j - 1)));
        xvel1(i, j) =
            xvel1(i, j) - stepbymass_s * (xarea(i, j) * (viscosity(i, j) - viscosity(i - 1, j + 0)) +
                                          xarea(i + 0, j - 1) * (viscosity(i + 0, j - 1) - viscosity(i - 1, j - 1)));
        yvel1(i, j) =
            yvel1(i, j) - stepbymass_s * (yarea(i, j) * (viscosity(i, j) - viscosity(i + 0, j - 1)) +
                                          yarea(i - 1, j + 0) * (viscosity(i - 1, j + 0) - viscosity(i - 1, j - 1)));
      });
}

//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    accelerate_kernel(globals.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt,
                      t.field.xarea, t.field.yarea, t.field.volume, t.field.density0, t.field.pressure,
                      t.field.viscosity, t.field.xvel0, t.field.yvel0, t.field.xvel1, t.field.yvel1);
  }

  if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;
}
