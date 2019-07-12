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
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		const AccDP2RW::View &density0,
		const AccDP2RW::View &energy0,
		const AccDP2RW::View &pressure,
		const AccDP2RW::View &viscosity,
		const AccDP2RW::View &soundspeed,
		const AccDP2RW::View &density1,
		const AccDP2RW::View &energy1,
		const AccDP2RW::View &xvel0,
		const AccDP2RW::View &yvel0,
		const AccDP2RW::View &xvel1,
		const AccDP2RW::View &yvel1,
		const AccDP2RW::View &vol_flux_x,
		const AccDP2RW::View &vol_flux_y,
		const AccDP2RW::View &mass_flux_x,
		const AccDP2RW::View &mass_flux_y,
		int left_xmin, int left_xmax, int left_ymin, int left_ymax,
		const AccDP2RW::View &left_density0,
		const AccDP2RW::View &left_energy0,
		const AccDP2RW::View &left_pressure,
		const AccDP2RW::View &left_viscosity,
		const AccDP2RW::View &left_soundspeed,
		const AccDP2RW::View &left_density1,
		const AccDP2RW::View &left_energy1,
		const AccDP2RW::View &left_xvel0,
		const AccDP2RW::View &left_yvel0,
		const AccDP2RW::View &left_xvel1,
		const AccDP2RW::View &left_yvel1,
		const AccDP2RW::View &left_vol_flux_x,
		const AccDP2RW::View &left_vol_flux_y,
		const AccDP2RW::View &left_mass_flux_x,
		const AccDP2RW::View &left_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_r_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		const AccDP2RW::View &density0,
		const AccDP2RW::View &energy0,
		const AccDP2RW::View &pressure,
		const AccDP2RW::View &viscosity,
		const AccDP2RW::View &soundspeed,
		const AccDP2RW::View &density1,
		const AccDP2RW::View &energy1,
		const AccDP2RW::View &xvel0,
		const AccDP2RW::View &yvel0,
		const AccDP2RW::View &xvel1,
		const AccDP2RW::View &yvel1,
		const AccDP2RW::View &vol_flux_x,
		const AccDP2RW::View &vol_flux_y,
		const AccDP2RW::View &mass_flux_x,
		const AccDP2RW::View &mass_flux_y,
		int right_xmin, int right_xmax, int right_ymin, int right_ymax,
		const AccDP2RW::View &right_density0,
		const AccDP2RW::View &right_energy0,
		const AccDP2RW::View &right_pressure,
		const AccDP2RW::View &right_viscosity,
		const AccDP2RW::View &right_soundspeed,
		const AccDP2RW::View &right_density1,
		const AccDP2RW::View &right_energy1,
		const AccDP2RW::View &right_xvel0,
		const AccDP2RW::View &right_yvel0,
		const AccDP2RW::View &right_xvel1,
		const AccDP2RW::View &right_yvel1,
		const AccDP2RW::View &right_vol_flux_x,
		const AccDP2RW::View &right_vol_flux_y,
		const AccDP2RW::View &right_mass_flux_x,
		const AccDP2RW::View &right_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

void update_tile_halo_t_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		const AccDP2RW::View &density0,
		const AccDP2RW::View &energy0,
		const AccDP2RW::View &pressure,
		const AccDP2RW::View &viscosity,
		const AccDP2RW::View &soundspeed,
		const AccDP2RW::View &density1,
		const AccDP2RW::View &energy1,
		const AccDP2RW::View &xvel0,
		const AccDP2RW::View &yvel0,
		const AccDP2RW::View &xvel1,
		const AccDP2RW::View &yvel1,
		const AccDP2RW::View &vol_flux_x,
		const AccDP2RW::View &vol_flux_y,
		const AccDP2RW::View &mass_flux_x,
		const AccDP2RW::View &mass_flux_y,
		int top_xmin, int top_xmax, int top_ymin, int top_ymax,
		const AccDP2RW::View &top_density0,
		const AccDP2RW::View &top_energy0,
		const AccDP2RW::View &top_pressure,
		const AccDP2RW::View &top_viscosity,
		const AccDP2RW::View &top_soundspeed,
		const AccDP2RW::View &top_density1,
		const AccDP2RW::View &top_energy1,
		const AccDP2RW::View &top_xvel0,
		const AccDP2RW::View &top_yvel0,
		const AccDP2RW::View &top_xvel1,
		const AccDP2RW::View &top_yvel1,
		const AccDP2RW::View &top_vol_flux_x,
		const AccDP2RW::View &top_vol_flux_y,
		const AccDP2RW::View &top_mass_flux_x,
		const AccDP2RW::View &top_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_b_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		const AccDP2RW::View &density0,
		const AccDP2RW::View &energy0,
		const AccDP2RW::View &pressure,
		const AccDP2RW::View &viscosity,
		const AccDP2RW::View &soundspeed,
		const AccDP2RW::View &density1,
		const AccDP2RW::View &energy1,
		const AccDP2RW::View &xvel0,
		const AccDP2RW::View &yvel0,
		const AccDP2RW::View &xvel1,
		const AccDP2RW::View &yvel1,
		const AccDP2RW::View &vol_flux_x,
		const AccDP2RW::View &vol_flux_y,
		const AccDP2RW::View &mass_flux_x,
		const AccDP2RW::View &mass_flux_y,
		int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
		const AccDP2RW::View &bottom_density0,
		const AccDP2RW::View &bottom_energy0,
		const AccDP2RW::View &bottom_pressure,
		const AccDP2RW::View &bottom_viscosity,
		const AccDP2RW::View &bottom_soundspeed,
		const AccDP2RW::View &bottom_density1,
		const AccDP2RW::View &bottom_energy1,
		const AccDP2RW::View &bottom_xvel0,
		const AccDP2RW::View &bottom_yvel0,
		const AccDP2RW::View &bottom_xvel1,
		const AccDP2RW::View &bottom_yvel1,
		const AccDP2RW::View &bottom_vol_flux_x,
		const AccDP2RW::View &bottom_vol_flux_y,
		const AccDP2RW::View &bottom_mass_flux_x,
		const AccDP2RW::View &bottom_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

#endif

