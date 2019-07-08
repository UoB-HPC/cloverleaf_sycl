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


#ifndef UPDATE_TILE_HALO_KERNEL_H
#define UPDATE_TILE_HALO_KERNEL_H

#include "definitions.h"

void update_tile_halo_l_kernel(
  int x_min, int x_max, int y_min,int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int left_xmin, int left_xmax, int left_ymin, int left_ymax,
  Kokkos::View<double**>& left_density0,
  Kokkos::View<double**>& left_energy0,
  Kokkos::View<double**>& left_pressure,
  Kokkos::View<double**>& left_viscosity,
  Kokkos::View<double**>& left_soundspeed,
  Kokkos::View<double**>& left_density1,
  Kokkos::View<double**>& left_energy1,
  Kokkos::View<double**>& left_xvel0,
  Kokkos::View<double**>& left_yvel0,
  Kokkos::View<double**>& left_xvel1,
  Kokkos::View<double**>& left_yvel1,
  Kokkos::View<double**>& left_vol_flux_x,
  Kokkos::View<double**>& left_vol_flux_y,
  Kokkos::View<double**>& left_mass_flux_x,
  Kokkos::View<double**>& left_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth);


void update_tile_halo_r_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int right_xmin, int right_xmax, int right_ymin, int right_ymax,
  Kokkos::View<double**>& right_density0,
  Kokkos::View<double**>& right_energy0,
  Kokkos::View<double**>& right_pressure,
  Kokkos::View<double**>& right_viscosity,
  Kokkos::View<double**>& right_soundspeed,
  Kokkos::View<double**>& right_density1,
  Kokkos::View<double**>& right_energy1,
  Kokkos::View<double**>& right_xvel0,
  Kokkos::View<double**>& right_yvel0,
  Kokkos::View<double**>& right_xvel1,
  Kokkos::View<double**>& right_yvel1,
  Kokkos::View<double**>& right_vol_flux_x,
  Kokkos::View<double**>& right_vol_flux_y,
  Kokkos::View<double**>& right_mass_flux_x,
  Kokkos::View<double**>& right_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth);

void update_tile_halo_t_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int top_xmin, int top_xmax, int top_ymin, int top_ymax,
  Kokkos::View<double**>& top_density0,
  Kokkos::View<double**>& top_energy0,
  Kokkos::View<double**>& top_pressure,
  Kokkos::View<double**>& top_viscosity,
  Kokkos::View<double**>& top_soundspeed,
  Kokkos::View<double**>& top_density1,
  Kokkos::View<double**>& top_energy1,
  Kokkos::View<double**>& top_xvel0,
  Kokkos::View<double**>& top_yvel0,
  Kokkos::View<double**>& top_xvel1,
  Kokkos::View<double**>& top_yvel1,
  Kokkos::View<double**>& top_vol_flux_x,
  Kokkos::View<double**>& top_vol_flux_y,
  Kokkos::View<double**>& top_mass_flux_x,
  Kokkos::View<double**>& top_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth);


void update_tile_halo_b_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
  Kokkos::View<double**>& bottom_density0,
  Kokkos::View<double**>& bottom_energy0,
  Kokkos::View<double**>& bottom_pressure,
  Kokkos::View<double**>& bottom_viscosity,
  Kokkos::View<double**>& bottom_soundspeed,
  Kokkos::View<double**>& bottom_density1,
  Kokkos::View<double**>& bottom_energy1,
  Kokkos::View<double**>& bottom_xvel0,
  Kokkos::View<double**>& bottom_yvel0,
  Kokkos::View<double**>& bottom_xvel1,
  Kokkos::View<double**>& bottom_yvel1,
  Kokkos::View<double**>& bottom_vol_flux_x,
  Kokkos::View<double**>& bottom_vol_flux_y,
  Kokkos::View<double**>& bottom_mass_flux_x,
  Kokkos::View<double**>& bottom_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth);

#endif

