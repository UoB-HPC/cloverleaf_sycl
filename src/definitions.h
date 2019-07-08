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

#ifndef GRID_H
#define GRID_H

#include <Kokkos_Core.hpp>

#define g_ibig 640000
#define g_small (1.0e-16)
#define g_big   (1.0e+21)
#define NUM_FIELDS 15

// Cannot call std::min or std::max from a CUDA kernel, so use these macros instead.
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) >= (b)) ? (a) : (b))


enum geometry_type {
	g_rect = 1, g_circ = 2, g_point = 3
};

// In the Fortran version these are 1,2,3,4,-1, but they are used firectly to index an array in this version
enum chunk_neighbour_type {
	chunk_left = 0, chunk_right = 1, chunk_bottom = 2, chunk_top = 3, external_face = -1
};
enum tile_neighbour_type {
	tile_left = 0, tile_right = 1, tile_bottom = 3, tile_top = 3, external_tile = -1
};

// Again, start at 0 as used for indexing an array of length NUM_FIELDS
enum field_parameter {

	field_density0 = 0,
	field_density1 = 1,
	field_energy0 = 2,
	field_energy1 = 3,
	field_pressure = 4,
	field_viscosity = 5,
	field_soundspeed = 6,
	field_xvel0 = 7,
	field_xvel1 = 8,
	field_yvel0 = 9,
	field_yvel1 = 10,
	field_vol_flux_x = 11,
	field_vol_flux_y = 12,
	field_mass_flux_x = 13,
	field_mass_flux_y = 14
};

enum data_parameter {
	cell_data = 1,
	vertex_data = 2,
	x_face_data = 3,
	y_face_data = 4
};

enum dir_parameter {
	g_xdir = 1, g_ydir = 2
};

struct state_type {

	bool defined;

	double density;
	double energy;
	double xvel;
	double yvel;

	geometry_type geometry;

	double xmin;
	double ymin;
	double xmax;
	double ymax;
	double radius;
};

struct grid_type {

	double xmin;
	double ymin;
	double xmax;
	double ymax;

	int x_cells;
	int y_cells;

};

struct profiler_type {

	double timestep;
	double acceleration;
	double PdV;
	double cell_advection;
	double mom_advection;
	double viscosity;
	double ideal_gas;
	double visit;
	double summary;
	double reset;
	double revert;
	double flux;
	double tile_halo_exchange;
	double self_halo_exchange;
	double mpi_halo_exchange;
};

struct field_type {

	Kokkos::View<double **> density0;
	Kokkos::View<double **> density1;
	Kokkos::View<double **> energy0;
	Kokkos::View<double **> energy1;
	Kokkos::View<double **> pressure;
	Kokkos::View<double **> viscosity;
	Kokkos::View<double **> soundspeed;
	Kokkos::View<double **> xvel0, xvel1;
	Kokkos::View<double **> yvel0, yvel1;
	Kokkos::View<double **> vol_flux_x, mass_flux_x;
	Kokkos::View<double **> vol_flux_y, mass_flux_y;

	Kokkos::View<double **> work_array1; // node_flux, stepbymass, volume_change, pre_vol
	Kokkos::View<double **> work_array2; // node_mass_post, post_vol
	Kokkos::View<double **> work_array3; // node_mass_pre,pre_mass
	Kokkos::View<double **> work_array4; // advec_vel, post_mass
	Kokkos::View<double **> work_array5; // mom_flux, advec_vol
	Kokkos::View<double **> work_array6; // pre_vol, post_ener
	Kokkos::View<double **> work_array7; // post_vol, ener_flux

	Kokkos::View<double *> cellx;
	Kokkos::View<double *> celly;
	Kokkos::View<double *> vertexx;
	Kokkos::View<double *> vertexy;
	Kokkos::View<double *> celldx;
	Kokkos::View<double *> celldy;
	Kokkos::View<double *> vertexdx;
	Kokkos::View<double *> vertexdy;

	Kokkos::View<double **> volume;
	Kokkos::View<double **> xarea;
	Kokkos::View<double **> yarea;

};

struct tile_type {

	field_type field;
	int tile_neighbours[4];
	int external_tile_mask[4];

	int t_xmin, t_xmax, t_ymin, t_ymax;

	int t_left, t_right, t_bottom, t_top;

};

struct chunk_type {

	int task; // MPI task

	int chunk_neighbours[4]; // Chunks, not tasks, so we can overload in the future

	// MPI Buffers in device memory
	Kokkos::View<double *> left_rcv_buffer, right_rcv_buffer, bottom_rcv_buffer, top_rcv_buffer;
	Kokkos::View<double *> left_snd_buffer, right_snd_buffer, bottom_snd_buffer, top_snd_buffer;

	// MPI Buffers in host memory - to be created with Kokkos::create_mirror_view() and Kokkos::deep_copy()
	typename Kokkos::View<double *>::HostMirror hm_left_rcv_buffer, hm_right_rcv_buffer, hm_bottom_rcv_buffer, hm_top_rcv_buffer;
	typename Kokkos::View<double *>::HostMirror hm_left_snd_buffer, hm_right_snd_buffer, hm_bottom_snd_buffer, hm_top_snd_buffer;

	tile_type *tiles;

	int x_min;
	int y_min;
	int x_max;
	int y_max;

	int left, right, bottom, top;
	int left_boundary, right_boundary, bottom_boundary, top_boundary;

};


// Collection of globally defined variables
struct global_variables {

	state_type *states;
	int number_of_states;

	int step;

	bool advect_x;

	int tiles_per_chunk;

	int error_condition;

	int test_problem;
	bool complete;

	bool profiler_on; // Internal code profiler to make comparisons accross systems easier

	profiler_type profiler;

	double end_time;

	int end_step;

	double dtold;
	double dt;
	double time;
	double dtinit;
	double dtmin;
	double dtmax;
	double dtrise;
	double dtu_safe;
	double dtv_safe;
	double dtc_safe;
	double dtdiv_safe;
	double dtc;
	double dtu;
	double dtv;
	double dtdiv;

	int visit_frequency;
	int summary_frequency;

	int jdt, kdt;

	chunk_type chunk;
	int number_of_chunks;

	grid_type grid;

};


#endif


