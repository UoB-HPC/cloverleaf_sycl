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


#include "viscosity.h"

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(handler &h, int x_min, int x_max, int y_min, int y_max,
                      AccDP1RW::Type celldx,
                      AccDP1RW::Type celldy,
                      AccDP2RW::Type density0,
                      AccDP2RW::Type pressure,
                      AccDP2RW::Type viscosity,
                      AccDP2RW::Type xvel0,
                      AccDP2RW::Type yvel0) {

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	par_ranged<class viscosity_>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](
			id<2> idx) {

		double ugrad = (xvel0[j<1>(idx)] + xvel0[jk<1, 1>(idx)]) - (xvel0[idx] + xvel0[k<1>(idx)]);

		double vgrad = (yvel0[k<1>(idx)] + yvel0[jk<1, 1>(idx)]) - (yvel0[idx] + yvel0[j<1>(idx)]);

		double div = (celldx[idx[0]] * (ugrad) + celldy[idx[1]] * (vgrad));

		double strain2 =
				0.5 * (xvel0[k<1>(idx)] + xvel0[jk<1, 1>(idx)] - xvel0[idx] - xvel0[j<1>(idx)]) /
				celldy[idx[1]]
				+ 0.5 * (yvel0[j<1>(idx)] + yvel0[jk<1, 1>(idx)] - yvel0[idx] - yvel0[k<1>(idx)]) /
				  celldx[idx[0]];

		double pgradx = (pressure[j<1>(idx)] - pressure[j<-1>(idx)]) /
		                (celldx[idx[0]] + celldx[idx[0] + 1]);
		double pgrady = (pressure[k<1>(idx)] - pressure[k<-1>(idx)]) /
		                (celldy[idx[1]] + celldy[idx[1] + 2]);

		double pgradx2 = pgradx * pgradx;
		double pgrady2 = pgrady * pgrady;

		double limiter =
				((0.5 * (ugrad) / celldx[idx[0]]) * pgradx2 +
				 (0.5 * (vgrad) / celldy[idx[1]]) * pgrady2 +
				 strain2 * pgradx * pgrady)
				/ MAX(pgradx2 + pgrady2, 1.0e-16);

		if ((limiter > 0.0) || (div >= 0.0)) {
			viscosity[idx] = 0.0;
		} else {
			double dirx = 1.0;
			if (pgradx < 0.0) dirx = -1.0;
			pgradx = dirx * MAX(1.0e-16, fabs(pgradx));
			double diry = 1.0;
			if (pgradx < 0.0) diry = -1.0;
			pgrady = diry * MAX(1.0e-16, fabs(pgrady));
			double pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
			double xgrad = fabs(celldx[idx[0]] * pgrad / pgradx);
			double ygrad = fabs(celldy[idx[1]] * pgrad / pgrady);
			double grad = MIN(xgrad, ygrad);
			double grad2 = grad * grad;

			viscosity[idx] = 2.0 * density0[idx] * grad2 * limiter * limiter;
		}
	});
}


//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial 
//  viscosity.
void viscosity(global_variables &globals) {

	execute(globals.queue, [&](handler &h) {

		for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
			tile_type &t = globals.chunk.tiles[tile];
			viscosity_kernel(h,
			                 t.info.t_xmin,
			                 t.info.t_xmax,
			                 t.info.t_ymin,
			                 t.info.t_ymax,
			                 t.field.celldx.access<RW>(h),
			                 t.field.celldy.access<RW>(h),
			                 t.field.density0.access<RW>(h),
			                 t.field.pressure.access<RW>(h),
			                 t.field.viscosity.access<RW>(h),
			                 t.field.xvel0.access<RW>(h),
			                 t.field.yvel0.access<RW>(h));
		}
	});

}

