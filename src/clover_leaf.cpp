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

//  @brief CloverLeaf top level program: Invokes the main cycle
//  @author Wayne Gaudin
//  @details CloverLeaf in a proxy-app that solves the compressible Euler
//  Equations using an explicit finite volume method on a Cartesian grid.
//  The grid is staggered with internal energy, density and pressure at cell
//  centres and velocities on cell vertices.
//
//  A second order predictor-corrector method is used to advance the solution
//  in time during the Lagrangian phase. A second order advective remap is then
//  carried out to return the mesh to an orthogonal state.
//
//  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
//  work on a mesh with varying spacing to keep it relevant to it's parent code.
//  For this reason, optimisations should only be carried out on the software
//  that do not change the underlying numerical method. For example, the
//  volume, though constant for all cells, should remain array and not be
//  converted to a scalar.

#include <mpi.h>

// #include <Kokkos_Core.hpp>

#include <iostream>

#include "comms.h"
#include "definitions.h"
#include "hydro.h"
#include "initialise.h"
#include "version.h"

// Output file handler
std::ostream g_out(nullptr);

int main(int argc, char *argv[]) {

  // Initialise MPI first
  MPI_Init(&argc, &argv);

  // Initialise Kokkos
  //	Kokkos::initialize();

  // Initialise communications
  struct parallel_ parallel;

  if (parallel.boss) {
    std::cout << std::endl
              << "Clover Version " << g_version << std::endl
              << "Kokkos Version" << std::endl
              << "Task Count " << parallel.max_task << std::endl
              << std::endl;
  }

  std::unique_ptr<global_variables> config = initialise(parallel, std::vector<std::string>(argv + 1, argv + argc));

  std::cout << "Launching hydro" << std::endl;
  hydro(*config, parallel);

  // Finilise programming models
  //	Kokkos::finalize();
  config->queue.wait_and_throw();
  MPI_Finalize();


  for (tile_type &t : config->chunk.tiles) {
    clover::free(config->queue,
                 t.field.density0, t.field.density1, t.field.energy0, t.field.energy1, t.field.pressure,
                 t.field.viscosity, t.field.soundspeed, t.field.xvel0, t.field.xvel1, t.field.yvel0, t.field.yvel1,
                 t.field.vol_flux_x, t.field.mass_flux_x, t.field.vol_flux_y, t.field.mass_flux_y,
                 t.field.work_array1, // node_flux, stepbymass, volume_change, pre_vol
                 t.field.work_array2, // node_mass_post, post_vol
                 t.field.work_array3, // node_mass_pre,pre_mass
                 t.field.work_array4, // advec_vel, post_mass
                 t.field.work_array5, // mom_flux, advec_vol
                 t.field.work_array6, // pre_vol, post_ener
                 t.field.work_array7, // post_vol, ener_flux
                 t.field.cellx, t.field.celldx, t.field.celly, t.field.celldy, t.field.vertexx, t.field.vertexdx,
                 t.field.vertexy, t.field.vertexdy, t.field.volume, t.field.xarea, t.field.yarea
    );
  }

  std::cout << "Done" << std::endl;
  return EXIT_SUCCESS;
}
