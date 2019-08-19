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


//  @brief Main set up routine
//  @author Wayne Gaudin
//  @details Invokes the mesh decomposer and sets up chunk connectivity. It then
//  allocates the communication buffers and call the chunk initialisation and
//  generation routines. It calls the equation of state to calculate initial
//  pressure before priming the halo cells and writing an initial field summary.


#include "start.h"
#include "build_field.h"
#include "initialise_chunk.h"
#include "generate_chunk.h"
#include "ideal_gas.h"
#include "field_summary.h"
#include "update_halo.h"
#include "visit.h"
#include "cxx14_compat.hpp"
#include <string>
#include <sstream>

extern std::ostream g_out;

void print_device(const cl::sycl::device &device) {
	auto exts = device.get_info<cl::sycl::info::device::extensions>();
	std::ostringstream extensions;
	std::copy(exts.begin(), exts.end(), std::ostream_iterator<std::string>(extensions, ","));

	auto type = device.get_info<cl::sycl::info::device::device_type>();
	auto typeName = "(unknown)";
	//@formatter:off
	switch (type){
		case sycl::info::device_type::cpu: typeName = "CPU"; break;
		case sycl::info::device_type::gpu: typeName = "GPU"; break;
		case sycl::info::device_type::accelerator: typeName = "ACCELERATOR"; break;
		case sycl::info::device_type::custom: typeName = "CUSTOM"; break;
		case sycl::info::device_type::automatic: typeName = "AUTOMATIC"; break;
		case sycl::info::device_type::host: typeName = "HOST"; break;
		case sycl::info::device_type::all: typeName = "ALL"; break;
	}
	//@formatter:on
	cl::sycl::platform platform = device.get_platform();
	std::cout << "[SYCL] Device        : " << device.get_info<cl::sycl::info::device::name>() << "\n";
	std::cout << "[SYCL]  - Type       : " << typeName << "\n";
	std::cout << "[SYCL]  - Vendor     : " << device.get_info<cl::sycl::info::device::vendor>() << "\n";
	std::cout << "[SYCL]  - Extensions : " << extensions.str() << "\n";
	std::cout << "[SYCL]  - Platform   : " << platform.get_info<cl::sycl::info::platform::name>() << "\n";
	std::cout << "[SYCL]     - Vendor  : " << platform.get_info<cl::sycl::info::platform::vendor>() << "\n";
	std::cout << "[SYCL]     - Version : " << platform.get_info<cl::sycl::info::platform::version>() << "\n";
	std::cout << "[SYCL]     - Profile : " << platform.get_info<cl::sycl::info::platform::profile>() << "\n";
}

std::unique_ptr<global_variables> start(parallel_ &parallel, const global_config &config) {

	if (parallel.boss) {
		g_out << "Setting up initial geometry" << std::endl
		      << std::endl;
	}

//	globals.time = 0.0;
//	globals.step = 0.0;
//	globals.dtold = globals.dtinit;
//	globals.dt = globals.dtinit;

	clover_barrier();

	// clover_get_num_chunks()

	int left, right, bottom, top;
	auto chunkNeighbours = clover_decompose(config, parallel,
	                                        config.grid.x_cells, config.grid.y_cells, left, right,
	                                        bottom, top);

	// Create the chunks
//	globals.chunk.task = parallel.task;

	int x_cells = right - left + 1;
	int y_cells = top - bottom + 1;


	auto handler = [](cl::sycl::exception_list exceptions) {
		for (std::exception_ptr const &e : exceptions) {
			try {
				std::rethrow_exception(e);
			} catch (cl::sycl::exception const &e) {
				std::cout << "[SYCL] Async exception:\n"
				          << e.what() << std::endl;
			}
		}
	};

	global_variables globals(config,
	                         cl::sycl::queue(cl::sycl::default_selector{}, handler),
	                         chunk_type(
			                         chunkNeighbours,
			                         parallel.task, 1, 1, x_cells, y_cells,
			                         left, right, bottom, top,
			                         1, config.grid.x_cells,
			                         1, config.grid.y_cells,
			                         config.tiles_per_chunk));


	auto devices = cl::sycl::device::get_devices();
	for(const auto& dev : devices) print_device(dev);

	std::cout << "[SYCL] Selected device:" << std::endl;
	print_device(globals.queue.get_device());


//	globals.chunk.left = left;
//	globals.chunk.bottom = bottom;
//	globals.chunk.right = right;
//	globals.chunk.top = top;
//	globals.chunk.left_boundary = 1;
//	globals.chunk.bottom_boundary = 1;
//	globals.chunk.right_boundary = globals.grid.x_cells;
//	globals.chunk.top_boundary = globals.grid.y_cells;
//	globals.chunk.x_min = 1;
//	globals.chunk.y_min = 1;
//	globals.chunk.x_max = x_cells;
//	globals.chunk.y_max = y_cells;

	if (DEBUG) std::cout << "Globals configured" << std::endl;

	auto infos = clover_tile_decompose(globals, x_cells, y_cells);

	std::transform(infos.begin(), infos.end(), std::back_inserter(globals.chunk.tiles),
	               [](const tile_info &ti) { return tile_type(ti); });



//	for(tile_type &tt : globals.chunk.tiles){
//		std::cout << tt.info.
//
//	}



	// Line 92 start.f90
	build_field(globals);

	clover_barrier(globals);

	clover_allocate_buffers(globals, parallel); // FIXME remove; basically no-op, moved to ctor

	if (parallel.boss) {
		g_out << "Generating chunks" << std::endl;
	}

	for (int tile = 0; tile < config.tiles_per_chunk; ++tile) {
		initialise_chunk(tile, globals);
		generate_chunk(tile, globals);
	}

//	globals.advect_x = true;

	clover_barrier(globals);

	// Do no profile the start up costs otherwise the total times will not add up
	// at the end
	bool profiler_off = globals.profiler_on;
	globals.profiler_on = false;

	for (int tile = 0; tile < config.tiles_per_chunk; ++tile) {
		ideal_gas(globals, tile, false);
	}

	// Prime all halo data for the first step
	// TODO replace with std::array
	int fields[NUM_FIELDS];
	for (int i = 0; i < NUM_FIELDS; ++i)
		fields[i] = 0;

	fields[field_density0] = 1;
	fields[field_energy0] = 1;
	fields[field_pressure] = 1;
	fields[field_viscosity] = 1;
	fields[field_density1] = 1;
	fields[field_energy1] = 1;
	fields[field_xvel0] = 1;
	fields[field_yvel0] = 1;
	fields[field_xvel1] = 1;
	fields[field_yvel1] = 1;

	update_halo(globals, fields, 2);

	if (parallel.boss) {
		g_out << std::endl
		      << "Problem initialised and generated" << std::endl;
	}

	field_summary(globals, parallel);

	if (config.visit_frequency != 0) visit(globals, parallel);

	clover_barrier(globals);

	globals.profiler_on = profiler_off;

	return make_unique<global_variables>(globals);
}

