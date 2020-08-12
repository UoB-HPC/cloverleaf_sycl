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

#define SPLIT


void calc_dt_kernel(
		queue &q,
		int x_min, int x_max, int y_min, int y_max,
		double dtmin,
		double dtc_safe,
		double dtu_safe,
		double dtv_safe,
		double dtdiv_safe,
		clover::Buffer<double, 2> xarea,
		clover::Buffer<double, 2> yarea,
		clover::Buffer<double, 1> cellx,
		clover::Buffer<double, 1> celly,
		clover::Buffer<double, 1> celldx,
		clover::Buffer<double, 1> celldy,
		clover::Buffer<double, 2> volume,
		clover::Buffer<double, 2> density0,
		clover::Buffer<double, 2> energy0,
		clover::Buffer<double, 2> pressure,
		clover::Buffer<double, 2> viscosity_a,
		clover::Buffer<double, 2> soundspeed,
		clover::Buffer<double, 2> xvel0,
		clover::Buffer<double, 2> yvel0,
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
//	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});


	auto policy = clover::Range2d(x_min + 1, y_min + 1, x_max + 2, y_max + 2);

#ifdef SPLIT


	clover::Buffer<double, 1> result(range<1>(policy.sizeX * policy.sizeY));

	clover::execute(q, [&](handler &h) {
		auto out = result.access<W>(h);
		auto xarea_ = xarea.access<R>(h);
		auto yarea_ = yarea.access<R>(h);
		auto celldx_ = celldx.access<R>(h);
		auto celldy_ = celldy.access<R>(h);
		auto volume_ = volume.access<R>(h);
		auto density0_ = density0.access<R>(h);
		auto viscosity_a_ = viscosity_a.access<R>(h);
		auto soundspeed_ = soundspeed.access<R>(h);
		auto xvel0_ = xvel0.access<R>(h);
		auto yvel0_ = yvel0.access<R>(h);

		clover::par_ranged<class APPEND_LN(advec_mom_x1)>(h, policy, [=](
				id<2> idx) {

			double dsx = celldx_[idx[0]];
			double dsy = celldy_[idx[1]];

			double cc = soundspeed_[idx] * soundspeed_[idx];
			cc = cc + 2.0 * viscosity_a_[idx] / density0_[idx];
			cc = sycl::fmax(sycl::sqrt(cc), g_small);

			double dtct = dtc_safe * sycl::fmin(dsx, dsy) / cc;

			double div = 0.0;

			double dv1 = (xvel0_[idx] + xvel0_[clover::offset(idx, 0, 1)]) * xarea_[idx];
			double dv2 =
					(xvel0_[clover::offset(idx, 1, 0)] + xvel0_[clover::offset(idx, 1, 1)]) *
					xarea_[clover::offset(
							idx, 1, 0)];

			div = div + dv2 - dv1;

			double dtut = dtu_safe * 2.0 * volume_[idx] /
			              sycl::fmax(sycl::fmax(
					              sycl::fabs(dv1), sycl::fabs(dv2)), g_small * volume_[idx]);

			dv1 = (yvel0_[idx] + yvel0_[clover::offset(idx, 1, 0)]) * yarea_[idx];
			dv2 = (yvel0_[clover::offset(idx, 0, 1)] + yvel0_[clover::offset(idx, 1, 1)]) *
			      yarea_[clover::offset(
					      idx, 0, 1)];

			div = div + dv2 - dv1;

			double dtvt = dtv_safe * 2.0 * volume_[idx] /
			              sycl::fmax(sycl::fmax(
					              sycl::fabs(dv1), sycl::fabs(dv2)), g_small * volume_[idx]);

			div = div / (2.0 * volume_[idx]);

			double dtdivt;
			if (div < -g_small) {
				dtdivt = dtdiv_safe * (-1.0 / div);
			} else {
				dtdivt = g_big;
			}

			double mins = sycl::fmin(dtct, sycl::fmin(dtut, sycl::fmin(dtvt, sycl::fmin(dtdivt, g_big))));

			size_t idx1d = (idx[0] - policy.fromX) * policy.sizeY + (idx[1] - policy.fromY);

			out[idx1d] = mins;

		});
	});


	struct captures {
		clover::Accessor<double, 1, R>::Type data;
	};
	typedef clover::local_reducer<double, double, captures> ctx;

	clover::par_reduce_1d<class dt_kernel_reduce, double>(
			q, clover::Range1d(0u, policy.sizeX * policy.sizeY),
			[=](handler &h, size_t &size) mutable {
				return ctx(h, size, {result.access<R>(h)}, result.buffer);
			},
			[](const ctx &ctx, id<1> lidx) { ctx.local[lidx] = g_big; },
			[](const ctx &ctx, id<1> lidx, id<1> idx) {
				ctx.local[lidx] = sycl::fmin(ctx.local[lidx], ctx.actual.data[idx]);
			},
			[](const ctx &ctx, id<1> idx, id<1> idy) { ctx.local[idx] = sycl::fmin(ctx.local[idx], ctx.local[idy]); },
			[](const ctx &ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });

	{
		dt_min_val = result.access<R>()[0];
	}

#else

	struct captures {
		clover::Accessor<double, 2, R>::Type xarea;
		clover::Accessor<double, 2, R>::Type yarea;
		clover::Accessor<double, 1, R>::Type celldx;
		clover::Accessor<double, 1, R>::Type celldy;
		clover::Accessor<double, 2, R>::Type volume;
		clover::Accessor<double, 2, R>::Type density0;
		clover::Accessor<double, 2, R>::Type viscosity_a;
		clover::Accessor<double, 2, R>::Type soundspeed;
		clover::Accessor<double, 2, R>::Type xvel0;
		clover::Accessor<double, 2, R>::Type yvel0;
	};


	typedef clover::local_reducer<double, double, captures> ctx;


	clover::Buffer<double, 1> result(range<1>(policy.sizeX * policy.sizeY));

	clover::par_reduce_2d<class dt_kernel_reduce, double>(
			q, policy,
			[=](handler &h, size_t &size) mutable {
				return ctx(h, size,
						   {xarea.access<R>(h),
							yarea.access<R>(h),
							celldx.access<R>(h),
							celldy.access<R>(h),
							volume.access<R>(h),
							density0.access<R>(h),
							viscosity_a.access<R>(h),
							soundspeed.access<R>(h),
							xvel0.access<R>(h),
							yvel0.access<R>(h) },
						   result.buffer);
			},
			[](const ctx &ctx, id<1> lidx) { ctx.local[lidx] = g_big; },
			[dtc_safe, dtv_safe, dtu_safe, dtdiv_safe](ctx ctx, id<1> lidx, id<2> idx) {


				double dsx = ctx.actual.celldx[idx[0]];
				double dsy = ctx.actual.celldy[idx[1]];

				double cc = ctx.actual.soundspeed[idx] * ctx.actual.soundspeed[idx];
				cc = cc + 2.0 * ctx.actual.viscosity_a[idx] / ctx.actual.density0[idx];
				cc = sycl::fmax(sycl::sqrt(cc), g_small);

				double dtct = dtc_safe * sycl::fmin(dsx, dsy) / cc;

				double div = 0.0;

				double dv1 = (ctx.actual.xvel0[idx] + ctx.actual.xvel0[clover::offset(idx, 0, 1)]) * ctx.actual.xarea[idx];
				double dv2 =
						(ctx.actual.xvel0[clover::offset(idx, 1, 0)] + ctx.actual.xvel0[clover::offset(idx, 1, 1)]) *
						ctx.actual.xarea[clover::offset(
								idx, 1, 0)];

				div = div + dv2 - dv1;

				double dtut = dtu_safe * 2.0 * ctx.actual.volume[idx] /
							  sycl::fmax(sycl::fmax(
									  sycl::fabs(dv1), sycl::fabs(dv2)), g_small * ctx.actual.volume[idx]);

				dv1 = (ctx.actual.yvel0[idx] + ctx.actual.yvel0[clover::offset(idx, 1, 0)]) * ctx.actual.yarea[idx];
				dv2 = (ctx.actual.yvel0[clover::offset(idx, 0, 1)] + ctx.actual.yvel0[clover::offset(idx, 1, 1)]) *
					  ctx.actual.yarea[clover::offset(
							  idx, 0, 1)];

				div = div + dv2 - dv1;

				double dtvt = dtv_safe * 2.0 * ctx.actual.volume[idx] /
							  sycl::fmax(sycl::fmax(
									  sycl::fabs(dv1), sycl::fabs(dv2)), g_small * ctx.actual.volume[idx]);

				div = div / (2.0 * ctx.actual.volume[idx]);

				double dtdivt;
				if (div < -g_small) {
					dtdivt = dtdiv_safe * (-1.0 / div);
				} else {
					dtdivt = g_big;
				}

				double mins = sycl::fmin(dtct, sycl::fmin(dtut, sycl::fmin(dtvt, sycl::fmin(dtdivt, g_big))));
				ctx.local[lidx] = sycl::fmin(ctx.local[lidx], mins);
			},
			[](const ctx &ctx, id<1> idx, id<1> idy) { ctx.local[idx] = sycl::fmin(ctx.local[idx], ctx.local[idy]); },
			[](const ctx &ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });

	{
		dt_min_val = result.access<R>()[0];
	}

#endif


	//  Extract the mimimum timestep information
	dtl_control = static_cast<int>(10.01 * (jk_control - static_cast<int>(jk_control)));
	jk_control = jk_control - (jk_control - (int) (jk_control));
	jldt = ((int) jk_control) % x_max;
	kldt = static_cast<int>(1.f + (jk_control / x_max));
	// TODO: cannot do this with GPU memory directly
	//xl_pos = cellx(jldt+1); // Offset by 1 because of Fortran halos in original code
	//yl_pos = celly(kldt+1);

	if (dt_min_val < dtmin) small = 1;


	if (small != 0) {

		auto cellx_acc = cellx.access<R>();
		auto celly_acc = celly.access<R>();
		auto density0_acc = density0.access<R>();
		auto energy0_acc = energy0.access<R>();
		auto pressure_acc = pressure.access<R>();
		auto soundspeed_acc = soundspeed.access<R>();
		auto xvel0_acc = xvel0.access<R>();
		auto yvel0_acc = yvel0.access<R>();

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

