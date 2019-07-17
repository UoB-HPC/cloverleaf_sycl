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
		AccDP2RW::Type density0,
		AccDP2RW::Type energy0,
		AccDP2RW::Type pressure,
		AccDP2RW::Type viscosity,
		AccDP2RW::Type soundspeed,
		AccDP2RW::Type density1,
		AccDP2RW::Type energy1,
		AccDP2RW::Type xvel0,
		AccDP2RW::Type yvel0,
		AccDP2RW::Type xvel1,
		AccDP2RW::Type yvel1,
		AccDP2RW::Type vol_flux_x,
		AccDP2RW::Type vol_flux_y,
		AccDP2RW::Type mass_flux_x,
		AccDP2RW::Type mass_flux_y,
		int left_xmin, int left_xmax, int left_ymin, int left_ymax,
		AccDP2RW::Type left_density0,
		AccDP2RW::Type left_energy0,
		AccDP2RW::Type left_pressure,
		AccDP2RW::Type left_viscosity,
		AccDP2RW::Type left_soundspeed,
		AccDP2RW::Type left_density1,
		AccDP2RW::Type left_energy1,
		AccDP2RW::Type left_xvel0,
		AccDP2RW::Type left_yvel0,
		AccDP2RW::Type left_xvel1,
		AccDP2RW::Type left_yvel1,
		AccDP2RW::Type left_vol_flux_x,
		AccDP2RW::Type left_vol_flux_y,
		AccDP2RW::Type left_mass_flux_x,
		AccDP2RW::Type left_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_r_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		AccDP2RW::Type density0,
		AccDP2RW::Type energy0,
		AccDP2RW::Type pressure,
		AccDP2RW::Type viscosity,
		AccDP2RW::Type soundspeed,
		AccDP2RW::Type density1,
		AccDP2RW::Type energy1,
		AccDP2RW::Type xvel0,
		AccDP2RW::Type yvel0,
		AccDP2RW::Type xvel1,
		AccDP2RW::Type yvel1,
		AccDP2RW::Type vol_flux_x,
		AccDP2RW::Type vol_flux_y,
		AccDP2RW::Type mass_flux_x,
		AccDP2RW::Type mass_flux_y,
		int right_xmin, int right_xmax, int right_ymin, int right_ymax,
		AccDP2RW::Type right_density0,
		AccDP2RW::Type right_energy0,
		AccDP2RW::Type right_pressure,
		AccDP2RW::Type right_viscosity,
		AccDP2RW::Type right_soundspeed,
		AccDP2RW::Type right_density1,
		AccDP2RW::Type right_energy1,
		AccDP2RW::Type right_xvel0,
		AccDP2RW::Type right_yvel0,
		AccDP2RW::Type right_xvel1,
		AccDP2RW::Type right_yvel1,
		AccDP2RW::Type right_vol_flux_x,
		AccDP2RW::Type right_vol_flux_y,
		AccDP2RW::Type right_mass_flux_x,
		AccDP2RW::Type right_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

void update_tile_halo_t_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		AccDP2RW::Type density0,
		AccDP2RW::Type energy0,
		AccDP2RW::Type pressure,
		AccDP2RW::Type viscosity,
		AccDP2RW::Type soundspeed,
		AccDP2RW::Type density1,
		AccDP2RW::Type energy1,
		AccDP2RW::Type xvel0,
		AccDP2RW::Type yvel0,
		AccDP2RW::Type xvel1,
		AccDP2RW::Type yvel1,
		AccDP2RW::Type vol_flux_x,
		AccDP2RW::Type vol_flux_y,
		AccDP2RW::Type mass_flux_x,
		AccDP2RW::Type mass_flux_y,
		int top_xmin, int top_xmax, int top_ymin, int top_ymax,
		AccDP2RW::Type top_density0,
		AccDP2RW::Type top_energy0,
		AccDP2RW::Type top_pressure,
		AccDP2RW::Type top_viscosity,
		AccDP2RW::Type top_soundspeed,
		AccDP2RW::Type top_density1,
		AccDP2RW::Type top_energy1,
		AccDP2RW::Type top_xvel0,
		AccDP2RW::Type top_yvel0,
		AccDP2RW::Type top_xvel1,
		AccDP2RW::Type top_yvel1,
		AccDP2RW::Type top_vol_flux_x,
		AccDP2RW::Type top_vol_flux_y,
		AccDP2RW::Type top_mass_flux_x,
		AccDP2RW::Type top_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);


void update_tile_halo_b_kernel(
		handler &h,
		int x_min, int x_max, int y_min, int y_max,
		AccDP2RW::Type density0,
		AccDP2RW::Type energy0,
		AccDP2RW::Type pressure,
		AccDP2RW::Type viscosity,
		AccDP2RW::Type soundspeed,
		AccDP2RW::Type density1,
		AccDP2RW::Type energy1,
		AccDP2RW::Type xvel0,
		AccDP2RW::Type yvel0,
		AccDP2RW::Type xvel1,
		AccDP2RW::Type yvel1,
		AccDP2RW::Type vol_flux_x,
		AccDP2RW::Type vol_flux_y,
		AccDP2RW::Type mass_flux_x,
		AccDP2RW::Type mass_flux_y,
		int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
		AccDP2RW::Type bottom_density0,
		AccDP2RW::Type bottom_energy0,
		AccDP2RW::Type bottom_pressure,
		AccDP2RW::Type bottom_viscosity,
		AccDP2RW::Type bottom_soundspeed,
		AccDP2RW::Type bottom_density1,
		AccDP2RW::Type bottom_energy1,
		AccDP2RW::Type bottom_xvel0,
		AccDP2RW::Type bottom_yvel0,
		AccDP2RW::Type bottom_xvel1,
		AccDP2RW::Type bottom_yvel1,
		AccDP2RW::Type bottom_vol_flux_x,
		AccDP2RW::Type bottom_vol_flux_y,
		AccDP2RW::Type bottom_mass_flux_x,
		AccDP2RW::Type bottom_mass_flux_y,
		int fields[NUM_FIELDS],
		int depth);

#endif

