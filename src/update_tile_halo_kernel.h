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
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		Accessor<double, 2, RW>::Type density0,
		Accessor<double, 2, RW>::Type energy0,
		Accessor<double, 2, RW>::Type pressure,
		Accessor<double, 2, RW>::Type viscosity,
		Accessor<double, 2, RW>::Type soundspeed,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type xvel0,
		Accessor<double, 2, RW>::Type yvel0,
		Accessor<double, 2, RW>::Type xvel1,
		Accessor<double, 2, RW>::Type yvel1,
		Accessor<double, 2, RW>::Type vol_flux_x,
		Accessor<double, 2, RW>::Type vol_flux_y,
		Accessor<double, 2, RW>::Type mass_flux_x,
		Accessor<double, 2, RW>::Type mass_flux_y,
		int left_xmin, int left_xmax, int left_ymin, int left_ymax,
		Accessor<double, 2, RW>::Type left_density0,
		Accessor<double, 2, RW>::Type left_energy0,
		Accessor<double, 2, RW>::Type left_pressure,
		Accessor<double, 2, RW>::Type left_viscosity,
		Accessor<double, 2, RW>::Type left_soundspeed,
		Accessor<double, 2, RW>::Type left_density1,
		Accessor<double, 2, RW>::Type left_energy1,
		Accessor<double, 2, RW>::Type left_xvel0,
		Accessor<double, 2, RW>::Type left_yvel0,
		Accessor<double, 2, RW>::Type left_xvel1,
		Accessor<double, 2, RW>::Type left_yvel1,
		Accessor<double, 2, RW>::Type left_vol_flux_x,
		Accessor<double, 2, RW>::Type left_vol_flux_y,
		Accessor<double, 2, RW>::Type left_mass_flux_x,
		Accessor<double, 2, RW>::Type left_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_r_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		Accessor<double, 2, RW>::Type density0,
		Accessor<double, 2, RW>::Type energy0,
		Accessor<double, 2, RW>::Type pressure,
		Accessor<double, 2, RW>::Type viscosity,
		Accessor<double, 2, RW>::Type soundspeed,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type xvel0,
		Accessor<double, 2, RW>::Type yvel0,
		Accessor<double, 2, RW>::Type xvel1,
		Accessor<double, 2, RW>::Type yvel1,
		Accessor<double, 2, RW>::Type vol_flux_x,
		Accessor<double, 2, RW>::Type vol_flux_y,
		Accessor<double, 2, RW>::Type mass_flux_x,
		Accessor<double, 2, RW>::Type mass_flux_y,
		int right_xmin, int right_xmax, int right_ymin, int right_ymax,
		Accessor<double, 2, RW>::Type right_density0,
		Accessor<double, 2, RW>::Type right_energy0,
		Accessor<double, 2, RW>::Type right_pressure,
		Accessor<double, 2, RW>::Type right_viscosity,
		Accessor<double, 2, RW>::Type right_soundspeed,
		Accessor<double, 2, RW>::Type right_density1,
		Accessor<double, 2, RW>::Type right_energy1,
		Accessor<double, 2, RW>::Type right_xvel0,
		Accessor<double, 2, RW>::Type right_yvel0,
		Accessor<double, 2, RW>::Type right_xvel1,
		Accessor<double, 2, RW>::Type right_yvel1,
		Accessor<double, 2, RW>::Type right_vol_flux_x,
		Accessor<double, 2, RW>::Type right_vol_flux_y,
		Accessor<double, 2, RW>::Type right_mass_flux_x,
		Accessor<double, 2, RW>::Type right_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

void update_tile_halo_t_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		Accessor<double, 2, RW>::Type density0,
		Accessor<double, 2, RW>::Type energy0,
		Accessor<double, 2, RW>::Type pressure,
		Accessor<double, 2, RW>::Type viscosity,
		Accessor<double, 2, RW>::Type soundspeed,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type xvel0,
		Accessor<double, 2, RW>::Type yvel0,
		Accessor<double, 2, RW>::Type xvel1,
		Accessor<double, 2, RW>::Type yvel1,
		Accessor<double, 2, RW>::Type vol_flux_x,
		Accessor<double, 2, RW>::Type vol_flux_y,
		Accessor<double, 2, RW>::Type mass_flux_x,
		Accessor<double, 2, RW>::Type mass_flux_y,
		int top_xmin, int top_xmax, int top_ymin, int top_ymax,
		Accessor<double, 2, RW>::Type top_density0,
		Accessor<double, 2, RW>::Type top_energy0,
		Accessor<double, 2, RW>::Type top_pressure,
		Accessor<double, 2, RW>::Type top_viscosity,
		Accessor<double, 2, RW>::Type top_soundspeed,
		Accessor<double, 2, RW>::Type top_density1,
		Accessor<double, 2, RW>::Type top_energy1,
		Accessor<double, 2, RW>::Type top_xvel0,
		Accessor<double, 2, RW>::Type top_yvel0,
		Accessor<double, 2, RW>::Type top_xvel1,
		Accessor<double, 2, RW>::Type top_yvel1,
		Accessor<double, 2, RW>::Type top_vol_flux_x,
		Accessor<double, 2, RW>::Type top_vol_flux_y,
		Accessor<double, 2, RW>::Type top_mass_flux_x,
		Accessor<double, 2, RW>::Type top_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_b_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		Accessor<double, 2, RW>::Type density0,
		Accessor<double, 2, RW>::Type energy0,
		Accessor<double, 2, RW>::Type pressure,
		Accessor<double, 2, RW>::Type viscosity,
		Accessor<double, 2, RW>::Type soundspeed,
		Accessor<double, 2, RW>::Type density1,
		Accessor<double, 2, RW>::Type energy1,
		Accessor<double, 2, RW>::Type xvel0,
		Accessor<double, 2, RW>::Type yvel0,
		Accessor<double, 2, RW>::Type xvel1,
		Accessor<double, 2, RW>::Type yvel1,
		Accessor<double, 2, RW>::Type vol_flux_x,
		Accessor<double, 2, RW>::Type vol_flux_y,
		Accessor<double, 2, RW>::Type mass_flux_x,
		Accessor<double, 2, RW>::Type mass_flux_y,
		int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
		Accessor<double, 2, RW>::Type bottom_density0,
		Accessor<double, 2, RW>::Type bottom_energy0,
		Accessor<double, 2, RW>::Type bottom_pressure,
		Accessor<double, 2, RW>::Type bottom_viscosity,
		Accessor<double, 2, RW>::Type bottom_soundspeed,
		Accessor<double, 2, RW>::Type bottom_density1,
		Accessor<double, 2, RW>::Type bottom_energy1,
		Accessor<double, 2, RW>::Type bottom_xvel0,
		Accessor<double, 2, RW>::Type bottom_yvel0,
		Accessor<double, 2, RW>::Type bottom_xvel1,
		Accessor<double, 2, RW>::Type bottom_yvel1,
		Accessor<double, 2, RW>::Type bottom_vol_flux_x,
		Accessor<double, 2, RW>::Type bottom_vol_flux_y,
		Accessor<double, 2, RW>::Type bottom_mass_flux_x,
		Accessor<double, 2, RW>::Type bottom_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

#endif

