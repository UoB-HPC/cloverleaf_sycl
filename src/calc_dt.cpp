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


#include "calc_dt.h"
#include "sycl_utils.hpp"
#include "sycl_reduction.hpp"

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.
void calc_dt_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		double dtmin,
		double dtc_safe,
		double dtu_safe,
		double dtv_safe,
		double dtdiv_safe,
		Buffer<double, 2> xarea,
		Buffer<double, 2> yarea,
		Buffer<double, 1> cellx,
		Buffer<double, 1> celly,
		Buffer<double, 1> celldx,
		Buffer<double, 1> celldy,
		Buffer<double, 2> volume,
		Buffer<double, 2> density0,
		Buffer<double, 2> energy0,
		Buffer<double, 2> pressure,
		Buffer<double, 2> viscosity_a,
		Buffer<double, 2> soundspeed,
		Buffer<double, 2> xvel0,
		Buffer<double, 2> yvel0,
		Buffer<double, 2> dt_min,
		double &dt_min_val,
		int &dtl_control,
		double &xl_pos,
		double &yl_pos,
		int &jldt,
		int &kldt,
		int &small) {


	small = 0;
	dt_min_val = g_big;
	double jk_control = 1.1;

	// DO k=y_min,y_max
	//   DO j=x_min,x_max
	// FIXME need par reduction for sycl
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

	struct captures {
		Accessor<double, 2, RW>::Type xarea;
		Accessor<double, 2, RW>::Type yarea;
		Accessor<double, 1, RW>::Type cellx;
		Accessor<double, 1, RW>::Type celly;
		Accessor<double, 1, RW>::Type celldx;
		Accessor<double, 1, RW>::Type celldy;
		Accessor<double, 2, RW>::Type volume;
		Accessor<double, 2, RW>::Type density0;
		Accessor<double, 2, RW>::Type energy0;
		Accessor<double, 2, RW>::Type pressure;
		Accessor<double, 2, RW>::Type viscosity_a;
		Accessor<double, 2, RW>::Type soundspeed;
		Accessor<double, 2, RW>::Type xvel0;
		Accessor<double, 2, RW>::Type yvel0;
		Accessor<double, 2, RW>::Type dt_min;
	};


	typedef local_reducer<double, double, captures> ctx;


	auto policy = Range2d(x_min + 1, y_min + 1, x_max + 2, y_max + 2);
	Buffer<double, 1> result(range<1>(policy.sizeX *policy.sizeY));


	par_reduce_2d<class dt_kernel_reduce>(
			q, policy,
			[=](handler &h, size_t &size) mutable {
				return ctx(h, size,
				           {xarea.access<RW>(h),
				            yarea.access<RW>(h),
				            cellx.access<RW>(h),
				            celly.access<RW>(h),
				            celldx.access<RW>(h),
				            celldy.access<RW>(h),
				            volume.access<RW>(h),
				            density0.access<RW>(h),
				            energy0.access<RW>(h),
				            pressure.access<RW>(h),
				            viscosity_a.access<RW>(h),
				            soundspeed.access<RW>(h),
				            xvel0.access<RW>(h),
				            yvel0.access<RW>(h),
				            dt_min.access<RW>(h)},
				           result.buffer);
			},
			[](ctx ctx, id<1> lidx) { ctx.local[lidx] = g_big; },
			[dtc_safe, dtv_safe, dtu_safe, dtdiv_safe](ctx ctx, id<1> lidx, id<2> idx) {


				double dsx = ctx.actual.celldx[idx[0]];
				double dsy = ctx.actual.celldy[idx[1]];

				double cc = ctx.actual.soundspeed[idx] * ctx.actual.soundspeed[idx];
				cc = cc + 2.0 * ctx.actual.viscosity_a[idx] / ctx.actual.density0[idx];
				cc = MAX(sqrt(cc), g_small);

				double dtct = dtc_safe * MIN(dsx, dsy) / cc;

				double div = 0.0;

				double dv1 = (ctx.actual.xvel0[idx] + ctx.actual.xvel0[k<1>(idx)]) * ctx.actual.xarea[idx];
				double dv2 =
						(ctx.actual.xvel0[j<1>(idx)] + ctx.actual.xvel0[jk<1, 1>(idx)]) * ctx.actual.xarea[j<1>(idx)];

				div = div + dv2 - dv1;

				double dtut = dtu_safe * 2.0 * ctx.actual.volume[idx] /
				              MAX(MAX(fabs(dv1), fabs(dv2)), g_small * ctx.actual.volume[idx]);

				dv1 = (ctx.actual.yvel0[idx] + ctx.actual.yvel0[j<1>(idx)]) * ctx.actual.yarea[idx];
				dv2 = (ctx.actual.yvel0[k<1>(idx)] + ctx.actual.yvel0[jk<1, 1>(idx)]) * ctx.actual.yarea[k<1>(idx)];

				div = div + dv2 - dv1;

				double dtvt = dtv_safe * 2.0 * ctx.actual.volume[idx] /
				              MAX(MAX(fabs(dv1), fabs(dv2)), g_small * ctx.actual.volume[idx]);

				div = div / (2.0 * ctx.actual.volume[idx]);

				double dtdivt;
				if (div < -g_small) {
					dtdivt = dtdiv_safe * (-1.0 / div);
				} else {
					dtdivt = g_big;
				}
				ctx.local[lidx] = MIN(dtct, MIN(dtut, MIN(dtvt, MIN(dtdivt, g_big))));
			},
			[](ctx ctx, id<1> idx, id<1> idy) { ctx.local[idx] = MIN(ctx.local[idx], ctx.local[idy]); },
			[](ctx ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });

	{
		std::cout << "V=" << dt_min_val << ";\n";
		dt_min_val = result.access<R>()[0];
		std::cout << "V=" << dt_min_val << ";\n";
	}



//	Kokkos::parallel_reduce("calc_dt", policy,
//	                        KOKKOS_LAMBDA(
//	const int j,
//	const int k,
//	double &dt_min_val) {
//
//		double dsx = celldx(j);
//		double dsy = celldy(k);
//
//		double cc = soundspeed(j, k) * soundspeed(j, k);
//		cc = cc + 2.0 * viscosity_a(j, k) / density0(j, k);
//		cc = MAX(sqrt(cc), g_small);
//
//		double dtct = dtc_safe * MIN(dsx, dsy) / cc;
//
//		double div = 0.0;
//
//		double dv1 = (xvel0(j, k) + xvel0(j, k + 1)) * xarea(j, k);
//		double dv2 = (xvel0(j + 1, k) + xvel0(j + 1, k + 1)) * xarea(j + 1, k);
//
//		div = div + dv2 - dv1;
//
//		double dtut = dtu_safe * 2.0 * volume(j, k) /
//		              MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume(j, k));
//
//		dv1 = (yvel0(j, k) + yvel0(j + 1, k)) * yarea(j, k);
//		dv2 = (yvel0(j, k + 1) + yvel0(j + 1, k + 1)) * yarea(j, k + 1);
//
//		div = div + dv2 - dv1;
//
//		double dtvt = dtv_safe * 2.0 * volume(j, k) /
//		              MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume(j, k));
//
//		div = div / (2.0 * volume(j, k));
//
//		double dtdivt;
//		if (div < -g_small) {
//			dtdivt = dtdiv_safe * (-1.0 / div);
//		} else {
//			dtdivt = g_big;
//		}
//
//		dt_min_val = MIN(dt_min_val, dtct);
//		dt_min_val = MIN(dt_min_val, dtut);
//		dt_min_val = MIN(dt_min_val, dtvt);
//		dt_min_val = MIN(dt_min_val, dtdivt);
//
//	},
//	Kokkos::Min<double>(dt_min_val));

//	par_ranged<class dt_min>(h, {0, 1}, [=](id<2>) {
//		// TODO remove
//	});


	//  Extract the mimimum timestep information
	dtl_control = 10.01 * (jk_control - (int) (jk_control));
	jk_control = jk_control - (jk_control - (int) (jk_control));
	jldt = ((int) jk_control) % x_max;
	kldt = 1 + (jk_control / x_max);
	// TODO: cannot do this with GPU memory directly
	//xl_pos = cellx(jldt+1); // Offset by 1 because of Fortran halos in original code
	//yl_pos = celly(kldt+1);

	if (dt_min_val < dtmin) small = 1;

//	cl::sycl::stream os(1024, 128, h);
//



	if (small != 0) {

		auto cellx_acc = cellx.access<RW>();
		auto celly_acc = celly.access<RW>();
		auto density0_acc = density0.access<RW>();
		auto energy0_acc = energy0.access<RW>();
		auto pressure_acc = pressure.access<RW>();
		auto soundspeed_acc = soundspeed.access<RW>();
		auto xvel0_acc = xvel0.access<RW>();
		auto yvel0_acc = yvel0.access<RW>();

		std::cout
				<< "Timestep information:" << std::endl
				<< "j, k                 : " << jldt << " " << kldt << std::endl
				<< "x, y                 : " << cellx_acc[jldt] << " " << celly_acc[kldt] << std::endl
				<< "timestep : " << dt_min_val << std::endl
				<< "Cell velocities;" << std::endl
				<< xvel0_acc[jldt][kldt] << " " << yvel0_acc[jldt][kldt] << std::endl
				<< xvel0_acc[jldt + 1][kldt] << " " << yvel0_acc[jldt + 1][kldt] << std::endl
				<< xvel0_acc[jldt + 1][kldt + 1] << " " << yvel0_acc[jldt + 1][kldt + 1] << std::endl
				<< xvel0_acc[jldt][kldt + 1] << " " << yvel0_acc[jldt][kldt + 1] << std::endl
				<< "density, energy, pressure, soundspeed " << std::endl
				<< density0_acc[jldt][kldt] << " " << energy0_acc[jldt][kldt] << " " << pressure_acc[jldt][kldt]
				<< " " << soundspeed_acc[jldt][kldt] << std::endl;
	}
}


//  @brief Driver for the timestep kernels
//  @author Wayne Gaudin
//  @details Invokes the user specified timestep kernel.
void calc_dt(global_variables &globals, int tile, double &local_dt, std::string &local_control,
             double &xl_pos, double &yl_pos, int &jldt, int &kldt) {

	local_dt = g_big;

	int l_control;
	int small = 0;


	tile_type &t = globals.chunk.tiles[tile];
	calc_dt_kernel(
			globals.queue,
			t.info.t_xmin,
			t.info.t_xmax,
			t.info.t_ymin,
			t.info.t_ymax,
			globals.config.dtmin,
			globals.config.dtc_safe,
			globals.config.dtu_safe,
			globals.config.dtv_safe,
			globals.config.dtdiv_safe,
			t.field.xarea,
			t.field.yarea,
			t.field.cellx,
			t.field.celly,
			t.field.celldx,
			t.field.celldy,
			t.field.volume,
			t.field.density0,
			t.field.energy0,
			t.field.pressure,
			t.field.viscosity,
			t.field.soundspeed,
			t.field.xvel0,
			t.field.yvel0,
			t.field.work_array1,
			local_dt,
			l_control,
			xl_pos,
			yl_pos,
			jldt,
			kldt,
			small
	);


	if (l_control == 1) local_control = "sound";
	if (l_control == 2) local_control = "xvel";
	if (l_control == 3) local_control = "yvel";
	if (l_control == 4) local_control = "div";

}

