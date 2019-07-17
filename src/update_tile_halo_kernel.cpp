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


#include "update_tile_halo_kernel.h"
#include "sycl_utils.hpp"

//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.



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
		int depth) {

	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_density0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				density0[x_min - j][k[0]] = left_density0[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_density1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				density1[x_min - j][k[0]] = left_density1[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_energy0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				energy0[x_min - j][k[0]] = left_energy0[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_energy1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				energy1[x_min - j][k[0]] = left_energy1[left_xmax + 1 - j][k[0]];
			}
		});
	}


	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_pressure>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				pressure[x_min - j][k[0]] = left_pressure[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_viscosity>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				viscosity[x_min - j][k[0]] = left_viscosity[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_soundspeed>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				soundspeed[x_min - j][k[0]] = left_soundspeed[left_xmax + 1 - j][k[0]];
			}
		});
	}


	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_xvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				xvel0[x_min - j][k[0]] = left_xvel0[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_xvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				xvel1[x_min - j][k[0]] = left_xvel1[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_yvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				yvel0[x_min - j][k[0]] = left_yvel0[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_yvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				yvel1[x_min - j][k[0]] = left_yvel1[left_xmax + 1 - j][k[0]];
			}
		});
	}


	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_vol_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				vol_flux_x[x_min - j][k[0]] = left_vol_flux_x[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_l_mass_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				mass_flux_x[x_min - j][k[0]] = left_mass_flux_x[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_vol_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				vol_flux_y[x_min - j][k[0]] = left_vol_flux_y[left_xmax + 1 - j][k[0]];
			}
		});
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_l_mass_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				mass_flux_y[x_min - j][k[0]] = left_mass_flux_y[left_xmax + 1 - j][k[0]];
			}
		});
	}

}

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
		int depth) {

	// Density 0
	if (fields[field_density0] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_density0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				density0[x_max + 2 + j][k[0]] = right_density0[right_xmin - 1 + 2 +
				                                               j][k[0]];
			}
		});
	}

	// Density 1
	if (fields[field_density1] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_density1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				density1[x_max + 2 + j][k[0]] = right_density1[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_energy0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				energy0[x_max + 2 + j][k[0]] = right_energy0[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_energy1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				energy1[x_max + 2 + j][k[0]] = right_energy1[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}


	// Pressure
	if (fields[field_pressure] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_pressure>(h, {y_min - depth + 1, y_max + depth + 2}, [=](

				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				pressure[x_max + 2 + j][k[0]] = right_pressure[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}

	// Viscosity
	if (fields[field_viscosity] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_viscosity>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				viscosity[x_max + 2 + j][k[0]] = right_viscosity[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_soundspeed>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				soundspeed[x_max + 2 + j][k[0]] = right_soundspeed[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}


	// XVEL 0
	if (fields[field_xvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_xvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				xvel0[x_max + 1 + 2 + j][k[0]] = right_xvel0[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_xvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				xvel1[x_max + 1 + 2 + j][k[0]] = right_xvel1[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_yvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				yvel0[x_max + 1 + 2 + j][k[0]] = right_yvel0[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_yvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				yvel1[x_max + 1 + 2 + j][k[0]] = right_yvel1[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}


	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_vol_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				vol_flux_x[x_max + 1 + 2 + j][k[0]] = right_vol_flux_x[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		// DO k=y_min-depth,y_max+depth
		par_ranged<class upd_halo_r_mass_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				mass_flux_x[x_max + 1 + 2 + j][k[0]] = right_mass_flux_x[right_xmin + 1 - 1 + 2 + j][k[0]];
			}
		});
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_vol_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				vol_flux_y[x_max + 2 + j][k[0]] = right_vol_flux_y[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		// DO k=y_min-depth,y_max+1+depth
		par_ranged<class upd_halo_r_mass_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
				id<1> k) {
			for (int j = 0; j < depth; ++j) {
				mass_flux_y[x_max + 2 + j][k[0]] = right_mass_flux_y[right_xmin - 1 + 2 + j][k[0]];
			}
		});
	}
}


//  Top and bottom only do xmin -> xmax
//  This is because the corner ghosts will get communicated in the left right communication

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
		int depth) {

	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_density0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				density0[j[0]][y_max + 2 + k] = top_density0[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_density1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				density1[j[0]][y_max + 2 + k] = top_density1[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_energy0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				energy0[j[0]][y_max + 2 + k] = top_energy0[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_energy1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				energy1[j[0]][y_max + 2 + k] = top_energy1[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}


	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_pressure>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				pressure[j[0]][y_max + 2 + k] = top_pressure[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_viscosity>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				viscosity[j[0]][y_max + 2 + k] = top_viscosity[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_soundspeed>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				soundspeed[j[0]][y_max + 2 + k] = top_soundspeed[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}


	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_xvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				xvel0[j[0]][y_max + 1 + 2 + k] = top_xvel0[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_xvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				xvel1[j[0]][y_max + 1 + 2 + k] = top_xvel1[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_yvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				yvel0[j[0]][y_max + 1 + 2 + k] = top_yvel0[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_yvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				yvel1[j[0]][y_max + 1 + 2 + k] = top_yvel1[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_vol_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				vol_flux_x[j[0]][y_max + 2 + k] = top_vol_flux_x[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_t_mass_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				mass_flux_x[j[0]][y_max + 2 + k] = top_mass_flux_x[j[0]][top_ymin - 1 + 2 + k];
			});
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_vol_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				vol_flux_y[j[0]][y_max + 1 + 2 + k] = top_vol_flux_y[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_t_mass_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				mass_flux_y[j[0]][y_max + 1 + 2 + k] = top_mass_flux_y[j[0]][top_ymin + 1 - 1 + 2 + k];
			});
		}
	}
}


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
		int depth) {

	// Density 0
	if (fields[field_density0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_density0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				density0[j[0]][y_min - k] = bottom_density0[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// Density 1
	if (fields[field_density1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_density1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				density1[j[0]][y_min - k] = bottom_density1[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// Energy 0
	if (fields[field_energy0] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_energy0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				energy0[j[0]][y_min - k] = bottom_energy0[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// Energy 1
	if (fields[field_energy1] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_energy1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				energy1[j[0]][y_min - k] = bottom_energy1[j[0]][bottom_ymax + 1 - k];
			});
		}
	}


	// Pressure
	if (fields[field_pressure] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_pressure>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				pressure[j[0]][y_min - k] = bottom_pressure[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// Viscocity
	if (fields[field_viscosity] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_viscosity>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				viscosity[j[0]][y_min - k] = bottom_viscosity[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// Soundspeed
	if (fields[field_soundspeed] == 1) {
		for (int k = 0; k < depth; ++k) {
			//  DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_soundspeed>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				soundspeed[j[0]][y_min - k] = bottom_soundspeed[j[0]][bottom_ymax + 1 - k];
			});
		}
	}


	// XVEL 0
	if (fields[field_xvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_xvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				xvel0[j[0]][y_min - k] = bottom_xvel0[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// XVEL 1
	if (fields[field_xvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_xvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				xvel1[j[0]][y_min - k] = bottom_xvel1[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// YVEL 0
	if (fields[field_yvel0] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_yvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				yvel0[j[0]][y_min - k] = bottom_yvel0[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// YVEL 1
	if (fields[field_yvel1] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_yvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				yvel1[j[0]][y_min - k] = bottom_yvel1[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// VOL_FLUX_X
	if (fields[field_vol_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_vol_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				vol_flux_x[j[0]][y_min - k] = bottom_vol_flux_x[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// MASS_FLUX_X
	if (fields[field_mass_flux_x] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+1+depth
			par_ranged<class upd_halo_b_mass_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
					id<1> j) {
				mass_flux_x[j[0]][y_min - k] = bottom_mass_flux_x[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// VOL_FLUX_Y
	if (fields[field_vol_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_vol_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				vol_flux_y[j[0]][y_min - k] = bottom_vol_flux_y[j[0]][bottom_ymax + 1 - k];
			});
		}
	}

	// MASS_FLUX_Y
	if (fields[field_mass_flux_y] == 1) {
		for (int k = 0; k < depth; ++k) {
			// DO j=x_min-depth, x_max+depth
			par_ranged<class upd_halo_b_mass_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
					id<1> j) {
				mass_flux_y[j[0]][y_min - k] = bottom_mass_flux_y[j[0]][bottom_ymax + 1 - k];
			});
		}
	}
}

