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


#include "comms.h"
#include "update_halo.h"
#include "update_tile_halo.h"
#include "timer.h"
#include "sycl_utils.hpp"


//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.
void update_halo_kernel(
		queue &queue,
		int x_min, int x_max, int y_min, int y_max,
		const std::array<int, 4> &chunk_neighbours,
		const std::array<int, 4> &tile_neighbours,
		field_type &field,
		int fields[NUM_FIELDS],
		int depth) {


	//  Update values in external halo cells based on depth and fields requested
	//  Even though half of these loops look the wrong way around, it should be noted
	//  that depth is either 1 or 2 so that it is more efficient to always thread
	//  loop along the mesh edge.
	if (fields[field_density0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density0 = field.density0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						density0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = density0[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density0 = field.density0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						density0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = density0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density0 = field.density0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						density0[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = density0[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density0 = field.density0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						density0[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = density0[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_density1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density1 = field.density1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						density1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = density1[cl::sycl::id<2>{j[0], static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density1 = field.density1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						density1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = density1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density1 = field.density1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						density1[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = density1[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto density1 = field.density1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						density1[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = density1[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_energy0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			//  DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy0 = field.energy0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						energy0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = energy0[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy0 = field.energy0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						energy0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = energy0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy0 = field.energy0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						energy0[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = energy0[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy0 = field.energy0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						energy0[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = energy0[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_energy1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy1 = field.energy1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						energy1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = energy1[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy1 = field.energy1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						energy1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = energy1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy1 = field.energy1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						energy1[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = energy1[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto energy1 = field.energy1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						energy1[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = energy1[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_pressure] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto pressure = field.pressure.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						pressure[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = pressure[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto pressure = field.pressure.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						pressure[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = pressure[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto pressure = field.pressure.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						pressure[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = pressure[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto pressure = field.pressure.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						pressure[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = pressure[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_viscosity] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto viscosity = field.viscosity.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						viscosity[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = viscosity[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto viscosity = field.viscosity.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						viscosity[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = viscosity[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto viscosity = field.viscosity.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						viscosity[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = viscosity[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto viscosity = field.viscosity.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						viscosity[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = viscosity[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_soundspeed] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto soundspeed = field.soundspeed.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						soundspeed[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = soundspeed[cl::sycl::id<2>{j[0],static_cast<size_t>(2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto soundspeed = field.soundspeed.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						soundspeed[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = soundspeed[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			//  DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto soundspeed = field.soundspeed.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						soundspeed[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = soundspeed[cl::sycl::id<2>{static_cast<size_t>(2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			//  DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto soundspeed = field.soundspeed.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						soundspeed[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = soundspeed[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_xvel0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel0 = field.xvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						xvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = xvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel0 = field.xvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						xvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 + 2 + k)}] = xvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel0 = field.xvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						xvel0[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = -xvel0[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel0 = field.xvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						xvel0[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + 1 + j),k[0]}] = -xvel0[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_xvel1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel1 = field.xvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						xvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = xvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel1 = field.xvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						xvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 + 2 + k)}] = xvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel1 = field.xvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						xvel1[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = -xvel1[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto xvel1 = field.xvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						xvel1[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + 1 + j),k[0]}] = -xvel1[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_yvel0] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel0 = field.yvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						yvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = -yvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel0 = field.yvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						yvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 + 2 + k)}] = -yvel0[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel0 = field.yvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						yvel0[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = yvel0[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel0 = field.yvel0.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						yvel0[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + 1 + j),k[0]}] = yvel0[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_yvel1] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel1 = field.yvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						yvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = -yvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel1 = field.yvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						yvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 + 2 + k)}] = -yvel1[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel1 = field.yvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						yvel1[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = yvel1[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto yvel1 = field.yvel1.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						yvel1[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + 1 + j),k[0]}] = yvel1[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_vol_flux_x] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_x = field.vol_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						vol_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = vol_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_x = field.vol_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						vol_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = vol_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_x = field.vol_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						vol_flux_x[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = -vol_flux_x[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_x = field.vol_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						vol_flux_x[cl::sycl::id<2>{static_cast<size_t>(x_max + j + 1 + 2),k[0]}] = -vol_flux_x[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_mass_flux_x] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_x = field.mass_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						mass_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = mass_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_x = field.mass_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						mass_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 2 + k)}] = mass_flux_x[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_x = field.mass_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						mass_flux_x[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = -mass_flux_x[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_x = field.mass_flux_x.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						mass_flux_x[cl::sycl::id<2>{static_cast<size_t>(x_max + j + 1 + 2),k[0]}] = -mass_flux_x[cl::sycl::id<2>{static_cast<size_t>(x_max + 1 - j),k[0]}];
					}
				});
			});
		}
	}


	if (fields[field_vol_flux_y] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_y = field.vol_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						vol_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = -vol_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_y = field.vol_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						vol_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + k + 1 + 2)}] = -vol_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_y = field.vol_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						vol_flux_y[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = vol_flux_y[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto vol_flux_y = field.vol_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						vol_flux_y[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = vol_flux_y[cl::sycl::id<2>{static_cast<size_t>(x_max - j),k[0]}];
					}
				});
			});
		}
	}

	if (fields[field_mass_flux_y] == 1) {
		if ((chunk_neighbours[chunk_bottom] == external_face) &&
		    (tile_neighbours[tile_bottom] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_y = field.mass_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						mass_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(1 - k)}] = -mass_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(1 + 2 + k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_top] == external_face) &&
		    (tile_neighbours[tile_top] == external_tile)) {
			// DO j=x_min-depth,x_max+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_y = field.mass_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](
						id<1> j) {
					for (int k = 0; k < depth; ++k) {
						mass_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + k + 1 + 2)}] = -mass_flux_y[cl::sycl::id<2>{j[0],static_cast<size_t>(y_max + 1 - k)}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_left] == external_face) &&
		    (tile_neighbours[tile_left] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_y = field.mass_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						mass_flux_y[cl::sycl::id<2>{static_cast<size_t>(1 - j),k[0]}] = mass_flux_y[cl::sycl::id<2>{static_cast<size_t>(1 + 2 + j),k[0]}];
					}
				});
			});
		}
		if ((chunk_neighbours[chunk_right] == external_face) &&
		    (tile_neighbours[tile_right] == external_tile)) {
			// DO k=y_min-depth,y_max+1+depth
			clover::execute(queue, [&](handler &h) {
				auto mass_flux_y = field.mass_flux_y.access<RW>(h);
				clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](
						id<1> k) {
					for (int j = 0; j < depth; ++j) {
						mass_flux_y[cl::sycl::id<2>{static_cast<size_t>(x_max + 2 + j),k[0]}] = mass_flux_y[cl::sycl::id<2>{static_cast<size_t>(x_max - j),k[0]}];
					}
				});
			});
		}
	}

}


//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_halo(global_variables &globals, int fields[NUM_FIELDS], const int depth) {

	double kernel_time;
	if (globals.profiler_on) kernel_time = timer();
	update_tile_halo(globals, fields, depth);
	if (globals.profiler_on) {
		globals.profiler.tile_halo_exchange += timer() - kernel_time;
		kernel_time = timer();
	}

	clover_exchange(globals, fields, depth);

	if (globals.profiler_on) {
		globals.profiler.mpi_halo_exchange += timer() - kernel_time;
		kernel_time = timer();
	}

	if ((globals.chunk.chunk_neighbours[chunk_left] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_right] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_bottom] == external_face) ||
	    (globals.chunk.chunk_neighbours[chunk_top] == external_face)) {


		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &t = globals.chunk.tiles[tile];
			update_halo_kernel(
					globals.queue,
					t.info.t_xmin,
					t.info.t_xmax,
					t.info.t_ymin,
					t.info.t_ymax,
					globals.chunk.chunk_neighbours,
					t.info.tile_neighbours,
					t.field,
					fields,
					depth);
		}


	}


	if (globals.profiler_on)
		globals.profiler.self_halo_exchange += timer() - kernel_time;
}

