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
#include "ideal_gas.h"
#include "sycl_utils.hpp"
#include "timer.h"

#include <iomanip>

extern std::ostream g_out;

//  @brief Fortran field summary kernel
//  @author Wayne Gaudin
//  @details The total mass, internal energy, kinetic energy and volume weighted
//  pressure for the chunk is calculated.
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

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }

  if (globals.profiler_on) {
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  struct summary {
    double vol = 0.0;
    double mass = 0.0;
    double ie = 0.0;
    double ke = 0.0;
    double press = 0.0;
    summary operator+(const summary &s) const {
      return {
          vol + s.vol, mass + s.mass, ie + s.ie, ke + s.ke, press + s.press,
      };
    }
  };

  clover::Buffer<summary, 1> summaryResults(1, globals.queue);
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    auto &field = t.field;

    globals.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(                                          //
              sycl::range<1>((ymax - ymin + 1) * (xmax - xmin + 1)), //
              sycl::reduction(summaryResults.data, {}, sycl::plus<>(),
                              sycl::property::reduction::initialize_to_identity()), //
              [=](sycl::id<1> idx, auto &acc) {
                const size_t j = xmin + 1 + idx[0] % (xmax - xmin + 1);
                const size_t k = ymin + 1 + idx[0] / (xmax - xmin + 1);

                double vsqrd = 0.0;
                for (size_t kv = k; kv <= k + 1; ++kv) {
                  for (size_t jv = j; jv <= j + 1; ++jv) {
                    vsqrd +=
                        0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
                  }
                }
                double cell_vol = field.volume(j, k);
                double cell_mass = cell_vol * field.density0(j, k);

                acc += summary{.vol = cell_vol,
                               .mass = cell_mass,
                               .ie = cell_mass * field.energy0(j, k),
                               .ke = cell_mass * 0.5 * vsqrd,
                               .press = cell_vol * field.pressure(j, k)};
              });
        })
        .wait_and_throw();
  }
  globals.queue.wait_and_throw();
  auto [vol, mass, ie, ke, press] = summaryResults[0];

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);

  if (globals.profiler_on) globals.profiler.summary += timer() - kernel_time;

  if (parallel.boss) {
    auto formatting = g_out.flags();
    g_out << " step: " << globals.step << std::scientific << std::setw(15) << vol << std::scientific << std::setw(15)
          << mass << std::scientific << std::setw(15) << mass / vol << std::scientific << std::setw(15) << press / vol
          << std::scientific << std::setw(15) << ie << std::scientific << std::setw(15) << ke << std::scientific
          << std::setw(15) << ie + ke << std::endl
          << std::endl;
    g_out.flags(formatting);
  }

  if (globals.complete) {
    double qa_diff;
    if (parallel.boss) {
      if (globals.config.test_problem >= 1) {
        if (globals.config.test_problem == 1) qa_diff = std::fabs((100.0 * (ke / 1.82280367310258)) - 100.0);
        if (globals.config.test_problem == 2) qa_diff = std::fabs((100.0 * (ke / 1.19316898756307)) - 100.0);
        if (globals.config.test_problem == 3) qa_diff = std::fabs((100.0 * (ke / 2.58984003503994)) - 100.0);
        if (globals.config.test_problem == 4) qa_diff = std::fabs((100.0 * (ke / 0.307475452287895)) - 100.0);
        if (globals.config.test_problem == 5) qa_diff = std::fabs((100.0 * (ke / 4.85350315783719)) - 100.0);
        std::cout << "Test problem " << globals.config.test_problem << " is within " << qa_diff
                  << "% of the expected solution" << std::endl;
        g_out << "Test problem " << globals.config.test_problem << " is within " << qa_diff
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
