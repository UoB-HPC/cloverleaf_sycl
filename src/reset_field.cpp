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

//  @brief Fortran reset field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of step filed data to the begining of
//  step data, ready for the next timestep.
void reset_field_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& yvel1) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy1({x_min+1, y_min+1}, {x_max+2, y_max+2});
  Kokkos::parallel_for("reset_field_1", policy1, KOKKOS_LAMBDA (const int j, const int k) {

    density0(j,k)=density1(j,k);
    energy0(j,k)=energy1(j,k);

  });

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy2({x_min+1, y_min+1}, {x_max+1+2, y_max+1+2});
  Kokkos::parallel_for("reset_field_2", policy2, KOKKOS_LAMBDA (const int j, const int k) {
    xvel0(j,k) = xvel1(j,k);
    yvel0(j,k) = yvel1(j,k);
  });

}


//  @brief Reset field driver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset kernel.
void reset_field(global_variables& globals) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    reset_field_kernel(
      globals.chunk.tiles[tile].t_xmin,
      globals.chunk.tiles[tile].t_xmax,
      globals.chunk.tiles[tile].t_ymin,
      globals.chunk.tiles[tile].t_ymax,
      globals.chunk.tiles[tile].field.density0,
      globals.chunk.tiles[tile].field.density1,
      globals.chunk.tiles[tile].field.energy0,
      globals.chunk.tiles[tile].field.energy1,
      globals.chunk.tiles[tile].field.xvel0,
      globals.chunk.tiles[tile].field.xvel1,
      globals.chunk.tiles[tile].field.yvel0,
      globals.chunk.tiles[tile].field.yvel1);
  }

    if (globals.profiler_on) globals.profiler.reset += timer()-kernel_time;
}

