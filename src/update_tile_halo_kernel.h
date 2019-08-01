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
#include "sycl_utils.hpp"

void update_tile_halo_l_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer<double, 2> &density0,
		clover::Buffer<double, 2> &energy0,
		clover::Buffer<double, 2> &pressure,
		clover::Buffer<double, 2> &viscosity,
		clover::Buffer<double, 2> &soundspeed,
		clover::Buffer<double, 2> &density1,
		clover::Buffer<double, 2> &energy1,
		clover::Buffer<double, 2> &xvel0,
		clover::Buffer<double, 2> &yvel0,
		clover::Buffer<double, 2> &xvel1,
		clover::Buffer<double, 2> &yvel1,
		clover::Buffer<double, 2> &vol_flux_x,
		clover::Buffer<double, 2> &vol_flux_y,
		clover::Buffer<double, 2> &mass_flux_x,
		clover::Buffer<double, 2> &mass_flux_y,
		int left_xmin, int left_xmax, int left_ymin, int left_ymax,
		clover::Buffer<double, 2> &left_density0,
		clover::Buffer<double, 2> &left_energy0,
		clover::Buffer<double, 2> &left_pressure,
		clover::Buffer<double, 2> &left_viscosity,
		clover::Buffer<double, 2> &left_soundspeed,
		clover::Buffer<double, 2> &left_density1,
		clover::Buffer<double, 2> &left_energy1,
		clover::Buffer<double, 2> &left_xvel0,
		clover::Buffer<double, 2> &left_yvel0,
		clover::Buffer<double, 2> &left_xvel1,
		clover::Buffer<double, 2> &left_yvel1,
		clover::Buffer<double, 2> &left_vol_flux_x,
		clover::Buffer<double, 2> &left_vol_flux_y,
		clover::Buffer<double, 2> &left_mass_flux_x,
		clover::Buffer<double, 2> &left_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_r_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer<double, 2> &density0,
		clover::Buffer<double, 2> &energy0,
		clover::Buffer<double, 2> &pressure,
		clover::Buffer<double, 2> &viscosity,
		clover::Buffer<double, 2> &soundspeed,
		clover::Buffer<double, 2> &density1,
		clover::Buffer<double, 2> &energy1,
		clover::Buffer<double, 2> &xvel0,
		clover::Buffer<double, 2> &yvel0,
		clover::Buffer<double, 2> &xvel1,
		clover::Buffer<double, 2> &yvel1,
		clover::Buffer<double, 2> &vol_flux_x,
		clover::Buffer<double, 2> &vol_flux_y,
		clover::Buffer<double, 2> &mass_flux_x,
		clover::Buffer<double, 2> &mass_flux_y,
		int right_xmin, int right_xmax, int right_ymin, int right_ymax,
		clover::Buffer<double, 2> &right_density0,
		clover::Buffer<double, 2> &right_energy0,
		clover::Buffer<double, 2> &right_pressure,
		clover::Buffer<double, 2> &right_viscosity,
		clover::Buffer<double, 2> &right_soundspeed,
		clover::Buffer<double, 2> &right_density1,
		clover::Buffer<double, 2> &right_energy1,
		clover::Buffer<double, 2> &right_xvel0,
		clover::Buffer<double, 2> &right_yvel0,
		clover::Buffer<double, 2> &right_xvel1,
		clover::Buffer<double, 2> &right_yvel1,
		clover::Buffer<double, 2> &right_vol_flux_x,
		clover::Buffer<double, 2> &right_vol_flux_y,
		clover::Buffer<double, 2> &right_mass_flux_x,
		clover::Buffer<double, 2> &right_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

void update_tile_halo_t_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer<double, 2> &density0,
		clover::Buffer<double, 2> &energy0,
		clover::Buffer<double, 2> &pressure,
		clover::Buffer<double, 2> &viscosity,
		clover::Buffer<double, 2> &soundspeed,
		clover::Buffer<double, 2> &density1,
		clover::Buffer<double, 2> &energy1,
		clover::Buffer<double, 2> &xvel0,
		clover::Buffer<double, 2> &yvel0,
		clover::Buffer<double, 2> &xvel1,
		clover::Buffer<double, 2> &yvel1,
		clover::Buffer<double, 2> &vol_flux_x,
		clover::Buffer<double, 2> &vol_flux_y,
		clover::Buffer<double, 2> &mass_flux_x,
		clover::Buffer<double, 2> &mass_flux_y,
		int top_xmin, int top_xmax, int top_ymin, int top_ymax,
		clover::Buffer<double, 2> &top_density0,
		clover::Buffer<double, 2> &top_energy0,
		clover::Buffer<double, 2> &top_pressure,
		clover::Buffer<double, 2> &top_viscosity,
		clover::Buffer<double, 2> &top_soundspeed,
		clover::Buffer<double, 2> &top_density1,
		clover::Buffer<double, 2> &top_energy1,
		clover::Buffer<double, 2> &top_xvel0,
		clover::Buffer<double, 2> &top_yvel0,
		clover::Buffer<double, 2> &top_xvel1,
		clover::Buffer<double, 2> &top_yvel1,
		clover::Buffer<double, 2> &top_vol_flux_x,
		clover::Buffer<double, 2> &top_vol_flux_y,
		clover::Buffer<double, 2> &top_mass_flux_x,
		clover::Buffer<double, 2> &top_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_b_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		clover::Buffer<double, 2> &density0,
		clover::Buffer<double, 2> &energy0,
		clover::Buffer<double, 2> &pressure,
		clover::Buffer<double, 2> &viscosity,
		clover::Buffer<double, 2> &soundspeed,
		clover::Buffer<double, 2> &density1,
		clover::Buffer<double, 2> &energy1,
		clover::Buffer<double, 2> &xvel0,
		clover::Buffer<double, 2> &yvel0,
		clover::Buffer<double, 2> &xvel1,
		clover::Buffer<double, 2> &yvel1,
		clover::Buffer<double, 2> &vol_flux_x,
		clover::Buffer<double, 2> &vol_flux_y,
		clover::Buffer<double, 2> &mass_flux_x,
		clover::Buffer<double, 2> &mass_flux_y,
		int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
		clover::Buffer<double, 2> &bottom_density0,
		clover::Buffer<double, 2> &bottom_energy0,
		clover::Buffer<double, 2> &bottom_pressure,
		clover::Buffer<double, 2> &bottom_viscosity,
		clover::Buffer<double, 2> &bottom_soundspeed,
		clover::Buffer<double, 2> &bottom_density1,
		clover::Buffer<double, 2> &bottom_energy1,
		clover::Buffer<double, 2> &bottom_xvel0,
		clover::Buffer<double, 2> &bottom_yvel0,
		clover::Buffer<double, 2> &bottom_xvel1,
		clover::Buffer<double, 2> &bottom_yvel1,
		clover::Buffer<double, 2> &bottom_vol_flux_x,
		clover::Buffer<double, 2> &bottom_vol_flux_y,
		clover::Buffer<double, 2> &bottom_mass_flux_x,
		clover::Buffer<double, 2> &bottom_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

#endif

