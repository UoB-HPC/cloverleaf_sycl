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

//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.



void update_tile_halo_l_kernel(
  int x_min, int x_max, int y_min,int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int left_xmin, int left_xmax, int left_ymin, int left_ymax,
  Kokkos::View<double**>& left_density0,
  Kokkos::View<double**>& left_energy0,
  Kokkos::View<double**>& left_pressure,
  Kokkos::View<double**>& left_viscosity,
  Kokkos::View<double**>& left_soundspeed,
  Kokkos::View<double**>& left_density1,
  Kokkos::View<double**>& left_energy1,
  Kokkos::View<double**>& left_xvel0,
  Kokkos::View<double**>& left_yvel0,
  Kokkos::View<double**>& left_xvel1,
  Kokkos::View<double**>& left_yvel1,
  Kokkos::View<double**>& left_vol_flux_x,
  Kokkos::View<double**>& left_vol_flux_y,
  Kokkos::View<double**>& left_mass_flux_x,
  Kokkos::View<double**>& left_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth) {

  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l density0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          density0(x_min-j,k)=left_density0(left_xmax+1-j,k);
        }
      });
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l density1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          density1(x_min-j,k)=left_density1(left_xmax+1-j,k);
        }
      });
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l energy0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          energy0(x_min-j,k)=left_energy0(left_xmax+1-j,k);
        }
      });
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l energy1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          energy1(x_min-j,k)=left_energy1(left_xmax+1-j,k);
        }
      });
  }


  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l pressure",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          pressure(x_min-j,k)=left_pressure(left_xmax+1-j,k);
        }
      });
  }
  
  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l viscosity",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          viscosity(x_min-j,k)=left_viscosity(left_xmax+1-j,k);
        }
      });
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l soundspeed",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          soundspeed(x_min-j,k)=left_soundspeed(left_xmax+1-j,k);
        }
      });
  }


  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l xvel0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          xvel0(x_min-j,k)=left_xvel0(left_xmax+1-j,k);
        }
      });
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l xvel1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          xvel1(x_min-j,k)=left_xvel1(left_xmax+1-j,k);
        }
      });
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l yvel0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          yvel0(x_min-j,k)=left_yvel0(left_xmax+1-j,k);
        }
      });
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l yvel1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          yvel1(x_min-j,k)=left_yvel1(left_xmax+1-j,k);
        }
      });
  }


  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l vol_flux_x",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_x(x_min-j,k)=left_vol_flux_x(left_xmax+1-j,k);
        }
      });
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_l mass_flux_x",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_x(x_min-j,k)=left_mass_flux_x(left_xmax+1-j,k);
        }
      });
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l vol_flux_y",
    Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
    KOKKOS_LAMBDA (const int k) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y(x_min-j,k)=left_vol_flux_y(left_xmax+1-j,k);
      }
    });
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_l mass_flux_y",
    Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
    KOKKOS_LAMBDA (const int k) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y(x_min-j,k)=left_mass_flux_y(left_xmax+1-j,k);
      }
    });
  }

}

void update_tile_halo_r_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int right_xmin, int right_xmax, int right_ymin, int right_ymax,
  Kokkos::View<double**>& right_density0,
  Kokkos::View<double**>& right_energy0,
  Kokkos::View<double**>& right_pressure,
  Kokkos::View<double**>& right_viscosity,
  Kokkos::View<double**>& right_soundspeed,
  Kokkos::View<double**>& right_density1,
  Kokkos::View<double**>& right_energy1,
  Kokkos::View<double**>& right_xvel0,
  Kokkos::View<double**>& right_yvel0,
  Kokkos::View<double**>& right_xvel1,
  Kokkos::View<double**>& right_yvel1,
  Kokkos::View<double**>& right_vol_flux_x,
  Kokkos::View<double**>& right_vol_flux_y,
  Kokkos::View<double**>& right_mass_flux_x,
  Kokkos::View<double**>& right_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth) {

  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r density0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          density0(x_max+2+j,k)=right_density0(right_xmin-1+2+j,k);
        }
      });
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r density1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          density1(x_max+2+j,k)=right_density1(right_xmin-1+2+j,k);
        }
      });
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r energy0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          energy0(x_max+2+j,k)=right_energy0(right_xmin-1+2+j,k);
        }
      });
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r energy1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          energy1(x_max+2+j,k)=right_energy1(right_xmin-1+2+j,k);
        }
      });
  }


  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r pressure",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          pressure(x_max+2+j,k)=right_pressure(right_xmin-1+2+j,k);
        }
      });
  }
  
  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r viscosity",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          viscosity(x_max+2+j,k)=right_viscosity(right_xmin-1+2+j,k);
        }
      });
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r soundspeed",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          soundspeed(x_max+2+j,k)=right_soundspeed(right_xmin-1+2+j,k);
        }
      });
  }


  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r xvel0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          xvel0(x_max+1+2+j,k)=right_xvel0(right_xmin+1-1+2+j,k);
        }
      });
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r xvel1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          xvel1(x_max+1+2+j,k)=right_xvel1(right_xmin+1-1+2+j,k);
        }
      });
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r yvel0",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          yvel0(x_max+1+2+j,k)=right_yvel0(right_xmin+1-1+2+j,k);
        }
      });
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r yvel1",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          yvel1(x_max+1+2+j,k)=right_yvel1(right_xmin+1-1+2+j,k);
        }
      });
  }


  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r vol_flux_x",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_x(x_max+1+2+j,k)=right_vol_flux_x(right_xmin+1-1+2+j,k);
        }
      });
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    Kokkos::parallel_for("update_tile_halo_r mass_flux_x",
      Kokkos::RangePolicy<>(y_min-depth+1, y_max+depth+2),
      KOKKOS_LAMBDA (const int k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_x(x_max+1+2+j,k)=right_mass_flux_x(right_xmin+1-1+2+j,k);
        }
      });
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r vol_flux_y",
    Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
    KOKKOS_LAMBDA (const int k) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y(x_max+2+j,k)=right_vol_flux_y(right_xmin-1+2+j,k);
      }
    });
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    Kokkos::parallel_for("update_tile_halo_r mass_flux_y",
    Kokkos::RangePolicy<>(y_min-depth+1, y_max+1+depth+2),
    KOKKOS_LAMBDA (const int k) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y(x_max+2+j,k)=right_mass_flux_y(right_xmin-1+2+j,k);
      }
    });
  }
}


//  Top and bottom only do xmin -> xmax
//  This is because the corner ghosts will get communicated in the left right communication

void update_tile_halo_t_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int top_xmin, int top_xmax, int top_ymin, int top_ymax,
  Kokkos::View<double**>& top_density0,
  Kokkos::View<double**>& top_energy0,
  Kokkos::View<double**>& top_pressure,
  Kokkos::View<double**>& top_viscosity,
  Kokkos::View<double**>& top_soundspeed,
  Kokkos::View<double**>& top_density1,
  Kokkos::View<double**>& top_energy1,
  Kokkos::View<double**>& top_xvel0,
  Kokkos::View<double**>& top_yvel0,
  Kokkos::View<double**>& top_xvel1,
  Kokkos::View<double**>& top_yvel1,
  Kokkos::View<double**>& top_vol_flux_x,
  Kokkos::View<double**>& top_vol_flux_y,
  Kokkos::View<double**>& top_mass_flux_x,
  Kokkos::View<double**>& top_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth) {

    // Density 0
    if (fields[field_density0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel density0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            density0(j,y_max+2+k)=top_density0(j,top_ymin-1+2+k);
          });
      }
    }

    // Density 1
    if (fields[field_density1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel density1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            density1(j,y_max+2+k)=top_density1(j,top_ymin-1+2+k);
          });
      }
    }

    // Energy 0
    if (fields[field_energy0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel energy0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            energy0(j,y_max+2+k)=top_energy0(j,top_ymin-1+2+k);
          });
      }
    }

    // Energy 1
    if (fields[field_energy1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel energy1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            energy1(j,y_max+2+k)=top_energy1(j,top_ymin-1+2+k);
          });
      }
    }
  

    // Pressure
    if (fields[field_pressure] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel pressure",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            pressure(j,y_max+2+k)=top_pressure(j,top_ymin-1+2+k);
          });
      }
    }

    // Viscocity
    if (fields[field_viscosity] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel viscosity",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            viscosity(j,y_max+2+k)=top_viscosity(j,top_ymin-1+2+k);
          });
      }
    }

    // Soundspeed
    if (fields[field_soundspeed] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel soundspeed",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            soundspeed(j,y_max+2+k)=top_soundspeed(j,top_ymin-1+2+k);
          });
      }
    }


    // XVEL 0
    if (fields[field_xvel0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel xvel0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            xvel0(j,y_max+1+2+k)=top_xvel0(j,top_ymin+1-1+2+k);
          });
      }
    }

    // XVEL 1
    if (fields[field_xvel1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel xvel1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            xvel1(j,y_max+1+2+k)=top_xvel1(j,top_ymin+1-1+2+k);
          });
      }
    }

    // YVEL 0
    if (fields[field_yvel0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel yvel0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            yvel0(j,y_max+1+2+k)=top_yvel0(j,top_ymin+1-1+2+k);
          });
      }
    }

    // YVEL 1
    if (fields[field_yvel1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel yvel1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            yvel1(j,y_max+1+2+k)=top_yvel1(j,top_ymin+1-1+2+k);
          });
      }
    }

    // VOL_FLUX_X
    if (fields[field_vol_flux_x] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel vol_flux_x",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            vol_flux_x(j,y_max+2+k)=top_vol_flux_x(j,top_ymin-1+2+k);
          });
      }
    }

    // MASS_FLUX_X
    if (fields[field_mass_flux_x] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel mass_flux_x",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            mass_flux_x(j,y_max+2+k)=top_mass_flux_x(j,top_ymin-1+2+k);
          });
      }
    }

    // VOL_FLUX_Y
    if (fields[field_vol_flux_y] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel vol_flux_y",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            vol_flux_y(j,y_max+1+2+k)=top_vol_flux_y(j,top_ymin+1-1+2+k);
          });
      }
    }

    // MASS_FLUX_Y
    if (fields[field_mass_flux_y] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel mass_flux_y",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            mass_flux_y(j,y_max+1+2+k)=top_mass_flux_y(j,top_ymin+1-1+2+k);
          });
      }
    }
}


void update_tile_halo_b_kernel(
  int x_min, int x_max, int y_min, int y_max,
  Kokkos::View<double**>& density0,
  Kokkos::View<double**>& energy0,
  Kokkos::View<double**>& pressure,
  Kokkos::View<double**>& viscosity,
  Kokkos::View<double**>& soundspeed,
  Kokkos::View<double**>& density1,
  Kokkos::View<double**>& energy1,
  Kokkos::View<double**>& xvel0,
  Kokkos::View<double**>& yvel0,
  Kokkos::View<double**>& xvel1,
  Kokkos::View<double**>& yvel1,
  Kokkos::View<double**>& vol_flux_x,
  Kokkos::View<double**>& vol_flux_y,
  Kokkos::View<double**>& mass_flux_x,
  Kokkos::View<double**>& mass_flux_y,
  int bottom_xmin, int bottom_xmax, int bottom_ymin, int bottom_ymax,
  Kokkos::View<double**>& bottom_density0,
  Kokkos::View<double**>& bottom_energy0,
  Kokkos::View<double**>& bottom_pressure,
  Kokkos::View<double**>& bottom_viscosity,
  Kokkos::View<double**>& bottom_soundspeed,
  Kokkos::View<double**>& bottom_density1,
  Kokkos::View<double**>& bottom_energy1,
  Kokkos::View<double**>& bottom_xvel0,
  Kokkos::View<double**>& bottom_yvel0,
  Kokkos::View<double**>& bottom_xvel1,
  Kokkos::View<double**>& bottom_yvel1,
  Kokkos::View<double**>& bottom_vol_flux_x,
  Kokkos::View<double**>& bottom_vol_flux_y,
  Kokkos::View<double**>& bottom_mass_flux_x,
  Kokkos::View<double**>& bottom_mass_flux_y,
  int fields[NUM_FIELDS],
  int depth) {

    // Density 0
    if (fields[field_density0] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel density0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            density0(j,y_min-k)=bottom_density0(j,bottom_ymax+1-k);
          });
      }
    }

    // Density 1
    if (fields[field_density1] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel density1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            density1(j,y_min-k)=bottom_density1(j,bottom_ymax+1-k);
          });
      }
    }

    // Energy 0
    if (fields[field_energy0] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel energy0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            energy0(j,y_min-k)=bottom_energy0(j,bottom_ymax+1-k);
          });
      }
    }

    // Energy 1
    if (fields[field_energy1] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel energy1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            energy1(j,y_min-k)=bottom_energy1(j,bottom_ymax+1-k);
          });
      }
    }
  

    // Pressure
    if (fields[field_pressure] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel pressure",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            pressure(j,y_min-k)=bottom_pressure(j,bottom_ymax+1-k);
          });
      }
    }

    // Viscocity
    if (fields[field_viscosity] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel viscosity",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            viscosity(j,y_min-k)=bottom_viscosity(j,bottom_ymax+1-k);
          });
      }
    }

    // Soundspeed
    if (fields[field_soundspeed] == 1) {
      for (int k = 0; k < depth; ++k) {
        //  DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel soundspeed",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            soundspeed(j,y_min-k)=bottom_soundspeed(j,bottom_ymax+1-k);
          });
      }
    }


    // XVEL 0
    if (fields[field_xvel0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel xvel0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            xvel0(j,y_min-k)=bottom_xvel0(j,bottom_ymax+1-k);
          });
      }
    }

    // XVEL 1
    if (fields[field_xvel1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel xvel1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            xvel1(j,y_min-k)=bottom_xvel1(j,bottom_ymax+1-k);
          });
      }
    }

    // YVEL 0
    if (fields[field_yvel0] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel yvel0",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            yvel0(j,y_min-k)=bottom_yvel0(j,bottom_ymax+1-k);
          });
      }
    }

    // YVEL 1
    if (fields[field_yvel1] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel yvel1",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            yvel1(j,y_min-k)=bottom_yvel1(j,bottom_ymax+1-k);
          });
      }
    }

    // VOL_FLUX_X
    if (fields[field_vol_flux_x] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel vol_flux_x",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            vol_flux_x(j,y_min-k)=bottom_vol_flux_x(j,bottom_ymax+1-k);
          });
      }
    }

    // MASS_FLUX_X
    if (fields[field_mass_flux_x] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+1+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel mass_flux_x",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+1+depth+2),
          KOKKOS_LAMBDA (const int j) {
            mass_flux_x(j,y_min-k)=bottom_mass_flux_x(j,bottom_ymax+1-k);
          });
      }
    }

    // VOL_FLUX_Y
    if (fields[field_vol_flux_y] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel vol_flux_y",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            vol_flux_y(j,y_min-k)=bottom_vol_flux_y(j,bottom_ymax+1-k);
          });
      }
    }

    // MASS_FLUX_Y
    if (fields[field_mass_flux_y] == 1) {
      for (int k = 0; k < depth; ++k) {
        // DO j=x_min-depth, x_max+depth
        Kokkos::parallel_for("update_tile_halo_t_kernel mass_flux_y",
          Kokkos::RangePolicy<>(x_min-depth+1, x_max+depth+2),
          KOKKOS_LAMBDA (const int j) {
            mass_flux_y(j,y_min-k)=bottom_mass_flux_y(j,bottom_ymax+1-k);
          });
      }
    }
}

