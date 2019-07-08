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


// @brief  Allocates the data for each mesh chunk
// @author Wayne Gaudin
// @details The data fields for the mesh chunk are allocated based on the mesh
// size.

#include "build_field.h"

// Allocate Kokkos Views for the data arrays
void build_field(global_variables& globals) {

  for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {

    const size_t xrange = (globals.chunk.tiles[tile].t_xmax+2) - (globals.chunk.tiles[tile].t_xmin-2) + 1;
    const size_t yrange = (globals.chunk.tiles[tile].t_ymax+2) - (globals.chunk.tiles[tile].t_ymin-2) + 1;

    // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
    new(&globals.chunk.tiles[tile].field.density0) Kokkos::View<double**>("density0", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.density1) Kokkos::View<double**>("density1", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.energy0) Kokkos::View<double**>("energy0", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.energy1) Kokkos::View<double**>("energy1", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.pressure) Kokkos::View<double**>("pressure", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.viscosity) Kokkos::View<double**>("viscosity", xrange, yrange);
    new(&globals.chunk.tiles[tile].field.soundspeed) Kokkos::View<double**>("soundspeed", xrange, yrange);

    // (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
    new(&globals.chunk.tiles[tile].field.xvel0) Kokkos::View<double**>("xvel0", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.xvel1) Kokkos::View<double**>("xvel1", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.yvel0) Kokkos::View<double**>("yvel0", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.yvel1) Kokkos::View<double**>("yvel1", xrange+1, yrange+1);

    // (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
    new(&globals.chunk.tiles[tile].field.vol_flux_x) Kokkos::View<double**>("vol_flux_x", xrange+1, yrange);
    new(&globals.chunk.tiles[tile].field.mass_flux_x) Kokkos::View<double**>("mass_flux_x", xrange+1, yrange);
    // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
    new(&globals.chunk.tiles[tile].field.vol_flux_y) Kokkos::View<double**>("vol_flux_y", xrange, yrange+1);
    new(&globals.chunk.tiles[tile].field.mass_flux_y) Kokkos::View<double**>("mass_flux_y", xrange, yrange+1);

    // (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
    new(&globals.chunk.tiles[tile].field.work_array1) Kokkos::View<double**>("work_array1", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array2) Kokkos::View<double**>("work_array2", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array3) Kokkos::View<double**>("work_array3", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array4) Kokkos::View<double**>("work_array4", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array5) Kokkos::View<double**>("work_array5", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array6) Kokkos::View<double**>("work_array6", xrange+1, yrange+1);
    new(&globals.chunk.tiles[tile].field.work_array7) Kokkos::View<double**>("work_array7", xrange+1, yrange+1);

    // (t_xmin-2:t_xmax+2)
    new(&globals.chunk.tiles[tile].field.cellx) Kokkos::View<double*>("cellx", xrange);
    new(&globals.chunk.tiles[tile].field.celldx) Kokkos::View<double*>("celldx", xrange);
    // (t_ymin-2:t_ymax+2)
    new(&globals.chunk.tiles[tile].field.celly) Kokkos::View<double*>("celly", yrange);
    new(&globals.chunk.tiles[tile].field.celldy) Kokkos::View<double*>("celldy", yrange);
    // (t_xmin-2:t_xmax+3)
    new(&globals.chunk.tiles[tile].field.vertexx) Kokkos::View<double*>("vertexx", xrange+1);
    new(&globals.chunk.tiles[tile].field.vertexdx) Kokkos::View<double*>("vertexdx", xrange+1);
    // (t_ymin-2:t_ymax+3)
    new(&globals.chunk.tiles[tile].field.vertexy) Kokkos::View<double*>("vertexy", yrange+1);
    new(&globals.chunk.tiles[tile].field.vertexdy) Kokkos::View<double*>("vertexdy", yrange+1);

    // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
    new(&globals.chunk.tiles[tile].field.volume) Kokkos::View<double**>("volume", xrange, yrange);
    // (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
    new(&globals.chunk.tiles[tile].field.xarea) Kokkos::View<double**>("xarea", xrange+1, yrange);
    // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
    new(&globals.chunk.tiles[tile].field.yarea) Kokkos::View<double**>("yarea", xrange, yrange+1);

    // Zeroing isn't strictly neccessary but it ensures physical pages
    // are allocated. This prevents first touch overheads in the main code
    // cycle which can skew timings in the first step

    // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
    field_type& field = globals.chunk.tiles[tile].field;

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+3) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_1({0,0}, {xrange+1,yrange+1});

    Kokkos::parallel_for("build_field_zero_1", loop_bounds_1, KOKKOS_LAMBDA (const int j, const int k) {

      field.work_array1(j,k) = 0.0;
      field.work_array2(j,k) = 0.0;
      field.work_array3(j,k) = 0.0;
      field.work_array4(j,k) = 0.0;
      field.work_array5(j,k) = 0.0;
      field.work_array6(j,k) = 0.0;
      field.work_array7(j,k) = 0.0;

      field.xvel0(j,k) = 0.0;
      field.xvel1(j,k) = 0.0;
      field.yvel0(j,k) = 0.0;
      field.yvel1(j,k) = 0.0;

    });

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+2) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_2({0,0}, {xrange,yrange});

    Kokkos::parallel_for("build_field_zero_2", loop_bounds_2, KOKKOS_LAMBDA (const int j, const int k) {

      field.density0(j,k) = 0.0;
      field.density1(j,k) = 0.0;
      field.energy0(j,k) = 0.0;
      field.energy1(j,k) = 0.0;
      field.pressure(j,k) = 0.0;
      field.viscosity(j,k) = 0.0;
      field.soundspeed(j,k) = 0.0;
      field.volume(j,k) = 0.0;

    });

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+3) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_3({0,0}, {xrange+1,yrange});

    Kokkos::parallel_for("build_field_zero_3", loop_bounds_3, KOKKOS_LAMBDA (const int j, const int k) {

      field.vol_flux_x(j,k) = 0.0;
      field.mass_flux_x(j,k) = 0.0;
      field.xarea(j,k) = 0.0;
    });

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+2) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_4({0,0}, {xrange,yrange+1});

    Kokkos::parallel_for("build_field_zero_4", loop_bounds_4, KOKKOS_LAMBDA (const int j, const int k) {

      field.vol_flux_y(j,k) = 0.0;
      field.mass_flux_y(j,k) = 0.0;
      field.yarea(j,k) = 0.0;
    });


    // (t_xmin-2:t_xmax+2) inclusive
    Kokkos::parallel_for("build_field_zero_5", xrange, KOKKOS_LAMBDA (const int j) {
      field.cellx(j) = 0.0;
      field.celldx(j) = 0.0;
    });

    // (t_ymin-2:t_ymax+2) inclusive
    Kokkos::parallel_for("build_field_zero_6", yrange, KOKKOS_LAMBDA (const int k) {
      field.celly(k) = 0.0;
      field.celldy(k) = 0.0;
    });

    // (t_xmin-2:t_xmax+3) inclusive
    Kokkos::parallel_for("build_field_zero_6", xrange+1, KOKKOS_LAMBDA (const int j) {
      field.vertexx(j) = 0.0;
      field.vertexdx(j) = 0.0;
    });

    // (t_ymin-2:t_ymax+3) inclusive
    Kokkos::parallel_for("build_field_zero_7", yrange+1, KOKKOS_LAMBDA (const int k) {
      field.vertexy(k) = 0.0;
      field.vertexdy(k) = 0.0;
    });

  }

}

