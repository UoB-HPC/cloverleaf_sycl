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


#include "field_summary.h"
#include "timer.h"
#include "ideal_gas.h"

#include <iomanip>

extern std::ostream g_out;

//  @brief Fortran field summary kernel
//  @author Wayne Gaudin
//  @details The total mass, internal energy, kinetic energy and volume weighted
//  pressure for the chunk is calculated.

// Kokkos cannot yet handle reductions over multiple variables using Lambda functions, but can using the functor version.
struct field_summary_functor {

	// Structure of variables to reduce
	typedef struct {
		double vol, mass, ie, ke, press;
	} value_type;

	// Functor data member (kernel arguments)
	int x_min, x_max, y_min, y_max;
	Kokkos::View<double **> volume;
	Kokkos::View<double **> density0;
	Kokkos::View<double **> energy0;
	Kokkos::View<double **> pressure;
	Kokkos::View<double **> xvel0;
	Kokkos::View<double **> yvel0;

	// Constructor, which saves the kernel arguments
	field_summary_functor(
			int x_min_, int x_max_, int y_min_, int y_max_,
			Kokkos::View<double **> volume_,
			Kokkos::View<double **> density0_,
			Kokkos::View<double **> energy0_,
			Kokkos::View<double **> pressure_,
			Kokkos::View<double **> xvel0_,
			Kokkos::View<double **> yvel0) :

			x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_),
			volume(volume_),
			density0(density0_),
			energy0(energy0_),
			pressure(pressure_),
			xvel0(xvel0_),
			yvel0(yvel0) {}

	// Kernel body
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, value_type &update) const {

		const int j = x_min + 1 + i % (x_max - x_min + 1);
		const int k = y_min + 1 + i / (x_max - x_min + 1);

		//
		// START OF THE KERNEL
		//

		double vsqrd = 0.0;
		for (int kv = k; kv <= k + 1; ++kv) {
			for (int jv = j; jv <= j + 1; ++jv) {
				vsqrd += 0.25 * (xvel0(jv, kv) * xvel0(jv, kv) + yvel0(jv, kv) * yvel0(jv, kv));
			}
		}
		double cell_vol = volume(j, k);
		double cell_mass = cell_vol * density0(j, k);
		update.vol += cell_vol;
		update.mass += cell_mass;
		update.ie += cell_mass * energy0(j, k);
		update.ke += cell_mass * 0.5 * vsqrd;
		update.press += cell_vol * pressure(j, k);

		//
		// END  OF THE KERNEL
		//

	};

	// Tell Kokkos how to reduce value_type
	KOKKOS_INLINE_FUNCTION
	void join(value_type &update, const value_type &input) const {
		update.vol += input.vol;
		update.mass += input.mass;
		update.ie += input.ie;
		update.ke += input.ke;
		update.press += input.press;
	}

	KOKKOS_INLINE_FUNCTION
	void join(volatile value_type &update, const volatile value_type &input) const {
		update.vol += input.vol;
		update.mass += input.mass;
		update.ie += input.ie;
		update.ke += input.ke;
		update.press += input.press;
	}

	// Initial values
	KOKKOS_INLINE_FUNCTION
	static void init(value_type &update) {
		update.vol = 0.0;
		update.mass = 0.0;
		update.ie = 0.0;
		update.ke = 0.0;
		update.press = 0.0;
	}
};


//  @brief Driver for the field summary kernels
//  @author Wayne Gaudin
//  @details The user specified field summary kernel is invoked here. A summation
//  across all mesh chunks is then performed and the information outputed.
//  If the run is a test problem, the final result is compared with the expected
//  result and the difference output.
//  Note the reference solution is the value returned from an Intel compiler with
//  ieee options set on a single core crun.

void field_summary(global_variables &globals, parallel_ &parallel) {

	if (parallel.boss) {
		g_out << std::endl
		      << "Time " << globals.time << std::endl
		      << "                "
		      << "Volume          "
		      << "Mass            "
		      << "Density         "
		      << "Pressure        "
		      << "Internal Energy "
		      << "Kinetic Energy  "
		      << "Total Energy    " << std::endl;
	}

	double kernel_time;
	if (globals.profiler_on) kernel_time = timer();

	for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {
		ideal_gas(globals, tile, false);
	}

	if (globals.profiler_on) {
		globals.profiler.ideal_gas += timer() - kernel_time;
		kernel_time = timer();
	}

	double vol = 0.0;
	double mass = 0.0;
	double ie = 0.0;
	double ke = 0.0;
	double press = 0.0;

	for (int tile = 0; tile < globals.tiles_per_chunk; ++tile) {
		field_summary_functor functor(
				globals.chunk.tiles[tile].t_xmin,
				globals.chunk.tiles[tile].t_xmax,
				globals.chunk.tiles[tile].t_ymin,
				globals.chunk.tiles[tile].t_ymax,
				globals.chunk.tiles[tile].field.volume,
				globals.chunk.tiles[tile].field.density0,
				globals.chunk.tiles[tile].field.energy0,
				globals.chunk.tiles[tile].field.pressure,
				globals.chunk.tiles[tile].field.xvel0,
				globals.chunk.tiles[tile].field.yvel0);

		typename field_summary_functor::value_type result;

		// Use a 1D parallel for because 2D reduction results in shared memory segfaults on a GPU
		Kokkos::parallel_reduce("field_summary",
		                        (globals.chunk.tiles[tile].t_ymax -
		                         globals.chunk.tiles[tile].t_ymin + 1) *
		                        (globals.chunk.tiles[tile].t_xmax -
		                         globals.chunk.tiles[tile].t_xmin + 1), functor, result);

		vol = result.vol;
		mass = result.mass;
		ie = result.ie;
		ke = result.ke;
		press = result.press;
	}

	clover_sum(vol);
	clover_sum(mass);
	clover_sum(ie);
	clover_sum(ke);
	clover_sum(press);

	if (globals.profiler_on) globals.profiler.summary += timer() - kernel_time;

	if (parallel.boss) {
		auto formatting = g_out.flags();
		g_out
				<< " step: " << globals.step
				<< std::scientific << std::setw(15) << vol
				<< std::scientific << std::setw(15) << mass
				<< std::scientific << std::setw(15) << mass / vol
				<< std::scientific << std::setw(15) << press / vol
				<< std::scientific << std::setw(15) << ie
				<< std::scientific << std::setw(15) << ke
				<< std::scientific << std::setw(15) << ie + ke << std::endl
				<< std::endl;
		g_out.flags(formatting);
	}

	if (globals.complete) {
		double qa_diff;
		if (parallel.boss) {
			if (globals.test_problem >= 1) {
				if (globals.test_problem == 1)
					qa_diff = fabs((100.0 * (ke / 1.82280367310258)) - 100.0);
				if (globals.test_problem == 2)
					qa_diff = fabs((100.0 * (ke / 1.19316898756307)) - 100.0);
				if (globals.test_problem == 3)
					qa_diff = fabs((100.0 * (ke / 2.58984003503994)) - 100.0);
				if (globals.test_problem == 4)
					qa_diff = fabs((100.0 * (ke / 0.307475452287895)) - 100.0);
				if (globals.test_problem == 5)
					qa_diff = fabs((100.0 * (ke / 4.85350315783719)) - 100.0);
				std::cout << "Test problem " << globals.test_problem << " is within " << qa_diff
				          << "% of the expected solution" << std::endl;
				g_out << "Test problem " << globals.test_problem << " is within " << qa_diff
				      << "% of the expected solution" << std::endl;
				if (qa_diff < 0.001) {
					std::cout << "This test is considered PASSED" << std::endl;
					g_out << "This test is considered PASSED" << std::endl;
				} else {
					std::cout << "This test is considered NOT PASSED" << std::endl;
					g_out << "This test is considered NOT PASSED" << std::endl;
				}
			}
		}
	}

}

