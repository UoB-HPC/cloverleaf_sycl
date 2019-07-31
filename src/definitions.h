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

#define DEBUG false

#define CL_TARGET_OPENCL_VERSION 220

#include <CL/sycl.hpp>
#include <iostream>
#include <utility>
#include "sycl_utils.hpp"

//#include <Kokkos_Core.hpp>

#define g_ibig 640000
#define g_small (1.0e-16)
#define g_big   (1.0e+21)
#define NUM_FIELDS 15

// Cannot call std::min or std::max from a CUDA kernel, so use these macros instead.


#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a ## b)
#define PP_CAT_II(p, res) res

#define APPEND_LN(base) PP_CAT(base, __LINE__)

using namespace cl;

using sycl::accessor;
using sycl::queue;
using sycl::buffer;
using sycl::range;
using sycl::handler;
using sycl::id;


constexpr sycl::access::mode R = sycl::access::mode::read;
constexpr sycl::access::mode W = sycl::access::mode::write;
constexpr sycl::access::mode RW = sycl::access::mode::read_write;



static inline id<2> offset(const id<2> idx, const int j, const int k) {
	int jj = static_cast<int>(idx[0]) + j;
	int kk = static_cast<int>(idx[1]) + k;
	return id<2>(jj, kk);
}

#include <fstream>
#include <iostream>

static inline void record(const std::string &name, const std::function<void(std::ofstream &)> &f) {
	std::ofstream out;
	out.open(name, std::ofstream::out | std::ofstream::trunc);
	f(out);
	out.close();
}

static inline void show(std::ostream &out, const std::string &name, Buffer<double, 1> &buffer) {
	auto view = buffer.template access<R>();
	const range<1> &range = view.get_range();
	out << name << "(" << 1 << ") [" << range[0] << "]" << std::endl;
	out << "\t";
	for (size_t i = 0; i < range[0]; ++i) {
		out << view[i] << ", ";
	}
	out << std::endl;
}

static inline void show(std::ostream &out, const std::string &name, Buffer<double, 2> &buffer) {
	auto view = buffer.access<R>();
	const range<2> &range = view.get_range();
	out << name << "(" << 2 << ") [" << range[0] << "x" << range[1] << "]" << std::endl;
	for (size_t i = 0; i < range[0]; ++i) {
		out << "\t";
		for (size_t j = 0; j < range[1]; ++j) {
			out << view[i][j] << ", ";
		}
		out << std::endl;
	}
}

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

	double timestep = 0.0;
	double acceleration = 0.0;
	double PdV = 0.0;
	double cell_advection = 0.0;
	double mom_advection = 0.0;
	double viscosity = 0.0;
	double ideal_gas = 0.0;
	double visit = 0.0;
	double summary = 0.0;
	double reset = 0.0;
	double revert = 0.0;
	double flux = 0.0;
	double tile_halo_exchange = 0.0;
	double self_halo_exchange = 0.0;
	double mpi_halo_exchange = 0.0;

};

struct field_type {

	Buffer<double, 2> density0;
	Buffer<double, 2> density1;
	Buffer<double, 2> energy0;
	Buffer<double, 2> energy1;
	Buffer<double, 2> pressure;
	Buffer<double, 2> viscosity;
	Buffer<double, 2> soundspeed;
	Buffer<double, 2> xvel0, xvel1;
	Buffer<double, 2> yvel0, yvel1;
	Buffer<double, 2> vol_flux_x, mass_flux_x;
	Buffer<double, 2> vol_flux_y, mass_flux_y;

	Buffer<double, 2> work_array1; // node_flux, stepbymass, volume_change, pre_vol
	Buffer<double, 2> work_array2; // node_mass_post, post_vol
	Buffer<double, 2> work_array3; // node_mass_pre,pre_mass
	Buffer<double, 2> work_array4; // advec_vel, post_mass
	Buffer<double, 2> work_array5; // mom_flux, advec_vol
	Buffer<double, 2> work_array6; // pre_vol, post_ener
	Buffer<double, 2> work_array7; // post_vol, ener_flux

	Buffer<double, 1> cellx;
	Buffer<double, 1> celldx;
	Buffer<double, 1> celly;
	Buffer<double, 1> celldy;
	Buffer<double, 1> vertexx;
	Buffer<double, 1> vertexdx;
	Buffer<double, 1> vertexy;
	Buffer<double, 1> vertexdy;

	Buffer<double, 2> volume;
	Buffer<double, 2> xarea;
	Buffer<double, 2> yarea;


	explicit field_type(const size_t xrange, const size_t yrange) :
			density0(range<2>(xrange, yrange)),
			density1(range<2>(xrange, yrange)),
			energy0(range<2>(xrange, yrange)),
			energy1(range<2>(xrange, yrange)),
			pressure(range<2>(xrange, yrange)),
			viscosity(range<2>(xrange, yrange)),
			soundspeed(range<2>(xrange, yrange)),
			xvel0(range<2>(xrange + 1, yrange + 1)),
			xvel1(range<2>(xrange + 1, yrange + 1)),
			yvel0(range<2>(xrange + 1, yrange + 1)),
			yvel1(range<2>(xrange + 1, yrange + 1)),
			vol_flux_x(range<2>(xrange + 1, yrange)),
			mass_flux_x(range<2>(xrange + 1, yrange)),
			vol_flux_y(range<2>(xrange, yrange + 1)),
			mass_flux_y(range<2>(xrange, yrange + 1)),
			work_array1(range<2>(xrange + 1, yrange + 1)),
			work_array2(range<2>(xrange + 1, yrange + 1)),
			work_array3(range<2>(xrange + 1, yrange + 1)),
			work_array4(range<2>(xrange + 1, yrange + 1)),
			work_array5(range<2>(xrange + 1, yrange + 1)),
			work_array6(range<2>(xrange + 1, yrange + 1)),
			work_array7(range<2>(xrange + 1, yrange + 1)),
			cellx(range<1>(xrange)),
			celldx(range<1>(xrange)),
			celly(range<1>(yrange)),
			celldy(range<1>(yrange)),
			vertexx(range<1>(xrange + 1)),
			vertexdx(range<1>(xrange + 1)),
			vertexy(range<1>(yrange + 1)),
			vertexdy(range<1>(yrange + 1)),
			volume(range<2>(xrange, yrange)),
			xarea(range<2>(xrange + 1, yrange)),
			yarea(range<2>(xrange, yrange + 1)) {}


};


struct tile_info {
	std::array<int, 4> tile_neighbours;
	std::array<int, 4> external_tile_mask;
	int t_xmin, t_xmax, t_ymin, t_ymax;
	int t_left, t_right, t_bottom, t_top;
};

struct tile_type {

//	tile_type(const size_t xrange, const size_t yrange) : field(xrange, yrange) {}

	tile_info info;
//	std::array<int, 4> tile_neighbours;
//	std::array<int, 4> external_tile_mask;
//	int t_xmin, t_xmax, t_ymin, t_ymax;
//	int t_left, t_right, t_bottom, t_top;

	field_type field;


	explicit tile_type(const tile_info &info) :
			info(info),
			// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
			// XXX see build_field()
			field((info.t_xmax + 2) - (info.t_xmin - 2) + 1,
			      (info.t_ymax + 2) - (info.t_ymin - 2) + 1) {}
};

struct chunk_type {

	// MPI Buffers in device memory

	// MPI Buffers in host memory - to be created with Kokkos::create_mirror_view() and Kokkos::deep_copy()
//	std::vector<double > hm_left_rcv_buffer, hm_right_rcv_buffer, hm_bottom_rcv_buffer, hm_top_rcv_buffer;
//	std::vector<double > hm_left_snd_buffer, hm_right_snd_buffer, hm_bottom_snd_buffer, hm_top_snd_buffer;
	const std::array<int, 4> chunk_neighbours; // Chunks, not tasks, so we can overload in the future

	const int task; // MPI task
	const int x_min;
	const int y_min;
	const int x_max;
	const int y_max;

	const int left, right, bottom, top;
	const int left_boundary, right_boundary, bottom_boundary, top_boundary;

	Buffer<double, 1> left_rcv_buffer, right_rcv_buffer, bottom_rcv_buffer, top_rcv_buffer;
	Buffer<double, 1> left_snd_buffer, right_snd_buffer, bottom_snd_buffer, top_snd_buffer;

	std::vector<tile_type> tiles;

	chunk_type(const std::array<int, 4> &chunkNeighbours,
	           const int task,
	           const int xMin, const int yMin, const int xMax, const int yMax,
	           const int left, const int right, const int bottom, const int top,
	           const int leftBoundary, const int rightBoundary, const int bottomBoundary, const int topBoundary,
	           const int tiles_per_chunk
	)
			: chunk_neighbours(chunkNeighbours),
			  task(task),
			  x_min(xMin), y_min(yMin), x_max(xMax), y_max(yMax),
			  left(left), right(right), bottom(bottom), top(top),
			  left_boundary(leftBoundary), right_boundary(rightBoundary),
			  bottom_boundary(bottomBoundary), top_boundary(topBoundary),
			  left_rcv_buffer(range<1>(10 * 2 * (yMax + 5))),
			  right_rcv_buffer(range<1>(10 * 2 * (yMax + 5))),
			  bottom_rcv_buffer(range<1>(10 * 2 * (xMax + 5))),
			  top_rcv_buffer(range<1>(10 * 2 * (xMax + 5))),
			  left_snd_buffer(range<1>(10 * 2 * (yMax + 5))),
			  right_snd_buffer(range<1>(10 * 2 * (yMax + 5))),
			  bottom_snd_buffer(range<1>(10 * 2 * (xMax + 5))),
			  top_snd_buffer(range<1>(10 * 2 * (xMax + 5))) {}


//	left_snd_buffer(range<1>(10 * 2 * (yMax + 5))),
//	left_rcv_buffer(range<1>(10 * 2 * (yMax + 5))),
//	right_snd_buffer(range<1>(10 * 2 * (yMax + 5))),
//	right_rcv_buffer(range<1>(10 * 2 * (yMax + 5))),
//	bottom_snd_buffer(range<1>(10 * 2 * (xMax + 5))),
//	bottom_rcv_buffer(range<1>(10 * 2 * (xMax + 5))),
//	top_snd_buffer(range<1>(10 * 2 * (xMax + 5))),
//	top_rcv_buffer(range<1>(10 * 2 * (xMax + 5)))

};


// Collection of globally defined variables
struct global_config {


	std::vector<state_type> states;


	int number_of_states;


	int tiles_per_chunk;


	int test_problem;

	bool profiler_on;


	double end_time;

	int end_step;

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


	int number_of_chunks;

	grid_type grid;

};

struct global_variables {

	const global_config config;

	cl::sycl::queue queue;
	chunk_type chunk;

	int error_condition;

	int step = 0;
	bool advect_x = true;
	double time = 0.0;

	double dt;
	double dtold;

	bool complete = false;
	int jdt, kdt;


	bool profiler_on; // Internal code profiler to make comparisons accross systems easier


	profiler_type profiler;


	explicit global_variables(
			const global_config &config,
			cl::sycl::queue queue,
			chunk_type chunk) :
			config(config), queue(std::move(queue)), chunk(std::move(chunk)),
			dt(config.dtinit),
			dtold(config.dtinit),
			profiler_on(config.profiler_on) {}


	void dump(const std::string &name) {

		std::cout << "Dumping globals to " << name  << std::endl;

		record(name, [&](std::ostream &out) {
			out << "Dump(tileCount = " << chunk.tiles.size() << ")" << std::endl;


			out << "error_condition" << '=' << error_condition << std::endl;

			out << "step" << '=' << step << std::endl;
			out << "advect_x" << '=' << advect_x << std::endl;
			out << "time" << '=' << time << std::endl;

			out << "dt" << '=' << dt << std::endl;
			out << "dtold" << '=' << dtold << std::endl;

			out << "complete" << '=' << complete << std::endl;
			out << "jdt" << '=' << jdt << std::endl;
			out << "kdt" << '=' << kdt << std::endl;


			for (size_t i = 0; i < chunk.tiles.size(); ++i) {
				auto fs = chunk.tiles[i].field;
				out << "\tTile[ " << i << "]:" << std::endl;

				tile_info &info = chunk.tiles[i].info;
				for (int l = 0; l < 4; ++l) {
					out << "info.tile_neighbours[i]" << '=' << info.tile_neighbours[i] << std::endl;
					out << "info.external_tile_mask[i]" << '=' << info.external_tile_mask[i] << std::endl;
				}

				out << "info.t_xmin" << '=' << info.t_xmin << std::endl;
				out << "info.t_xmax" << '=' << info.t_xmax << std::endl;
				out << "info.t_ymin" << '=' << info.t_ymin << std::endl;
				out << "info.t_ymax" << '=' << info.t_ymax << std::endl;
				out << "info.t_left" << '=' << info.t_left << std::endl;
				out << "info.t_right" << '=' << info.t_right << std::endl;
				out << "info.t_bottom" << '=' << info.t_bottom << std::endl;
				out << "info.t_top" << '=' << info.t_top << std::endl;


				show(out, "density0", fs.density0);
				show(out, "density1", fs.density1);
				show(out, "energy0", fs.energy0);
				show(out, "energy1", fs.energy1);
				show(out, "pressure", fs.pressure);
				show(out, "viscosity", fs.viscosity);
				show(out, "soundspeed", fs.soundspeed);
				show(out, "xvel0", fs.xvel0);
				show(out, "xvel1", fs.xvel1);
				show(out, "yvel0", fs.yvel0);
				show(out, "yvel1", fs.yvel1);
				show(out, "vol_flux_x", fs.vol_flux_x);
				show(out, "vol_flux_y", fs.vol_flux_y);
				show(out, "mass_flux_x", fs.mass_flux_x);
				show(out, "mass_flux_y", fs.mass_flux_y);

				show(out, "work_array1", fs.work_array1); // node_flux, stepbymass, volume_change, pre_vol
				show(out, "work_array2", fs.work_array2); // node_mass_post, post_vol
				show(out, "work_array3", fs.work_array3); // node_mass_pre,pre_mass
				show(out, "work_array4", fs.work_array4); // advec_vel, post_mass
				show(out, "work_array5", fs.work_array5); // mom_flux, advec_vol
				show(out, "work_array6", fs.work_array6); // pre_vol, post_ener
				show(out, "work_array7", fs.work_array7); // post_vol, ener_flux

				show(out, "cellx", fs.cellx);
				show(out, "celldx", fs.celldx);
				show(out, "celly", fs.celly);
				show(out, "celldy", fs.celldy);
				show(out, "vertexx", fs.vertexx);
				show(out, "vertexdx", fs.vertexdx);
				show(out, "vertexy", fs.vertexy);
				show(out, "vertexdy", fs.vertexdy);

				show(out, "volume", fs.volume);
				show(out, "xarea", fs.xarea);
				show(out, "yarea", fs.yarea);
			}
		});

	}

};


#endif


