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

#ifndef CLOVERLEAF_SYCL_SYCL_REDUCTION_HPP
#define CLOVERLEAF_SYCL_SYCL_REDUCTION_HPP

#include "sycl_utils.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <utility>

namespace clover {

template <typename T, typename U, typename C> struct local_reducer {

  sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local;

  C actual;
  sycl::accessor<U, 1, sycl::access::mode::read_write, sycl::access::target::device> result;

  local_reducer(sycl::handler &h, size_t size, C actual, sycl::buffer<U, 1> &b)
      : local(sycl::range<1>(size), h), actual(actual),
        result(b.template get_access<sycl::access::mode::read_write>(h)) {}

  inline void drain(sycl::id<1> lid, sycl::id<1> gid) const { local[lid] = result[gid]; }
};

template <typename nameT, size_t dimension, class RangeTpe, class LocalType, class LocalAllocator = std::nullptr_t,
          class Empty = std::nullptr_t, class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t,
          class Finaliser = std::nullptr_t, class RangeLengthFn = std::nullptr_t, class RangeIdFn = std::nullptr_t>
static inline void par_reduce_nd_impl(sycl::queue &q, RangeTpe range, sycl::buffer<LocalType, 1> result,
                                      RangeLengthFn lengthFn, RangeIdFn rangeIdFn, LocalAllocator allocator,
                                      const LocalType identity, Functor functor, BinaryOp combiner) {

  auto dev = q.get_device();

  size_t dot_num_groups;
  size_t dot_wgsize;
  if (dev.is_cpu()) {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>();
    dot_wgsize = dev.get_info<sycl::info::device::native_vector_width_double>() * 2;

  } else {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>() * 4;
    // TODO: not sure about this, max reduction wg_size = 512
    dot_wgsize = dev.get_info<sycl::info::device::max_work_group_size>();
    dot_wgsize = dot_wgsize > 512 ? 512 : dot_wgsize;
  }

  size_t N = lengthFn(range);
  dot_num_groups = std::min(N, dot_num_groups);
#ifdef SYCL_DEBUG
  std::cout << "RD: dot_wgsize=" << dot_wgsize << " dot_num_groups:" << dot_num_groups << " N=" << N << "\n";
#endif

  q.submit([=](sycl::handler &h) mutable {
    auto ctx = allocator(h, dot_wgsize);
    auto reduction = sycl::reduction(result, h, identity, combiner);
    h.parallel_for<nameT>(sycl::nd_range<1>(dot_num_groups * dot_wgsize, dot_wgsize), reduction,
                          [=](sycl::nd_item<1> item, auto &red_sum) {

#ifdef USE_PRE_SYCL121R3
                            size_t i = item.get_global(0);
                            size_t li = item.get_local(0);
#else
						size_t i = item.get_global_id(0);
						size_t li = item.get_local_id(0);
#endif

                            size_t global_size = item.get_global_range()[0];

                            for (; i < N; i += global_size) {
                              functor(ctx, rangeIdFn(sycl::id<1>(i), range), red_sum);
                            }
                          });
  });

#ifdef SYNC_KERNELS
  q.wait_and_throw();
#endif
#ifdef SYCL_DEBUG
  std::cout << "RD: done= " << N << "\n";
#endif
}

// applies a 1d reduction
template <typename nameT, class LocalType, class LocalAllocator = std::nullptr_t, class Empty = std::nullptr_t,
          class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t, class Finaliser = std::nullptr_t>
static inline void par_reduce_1d(sycl::queue &q, const clover::Range1d &range, sycl::buffer<LocalType, 1> result,
                                 LocalAllocator allocator, const LocalType identity, Functor functor,
                                 BinaryOp combiner) {
#ifdef SYCL_DEBUG
  std::cout << "par_reduce_1d " << range << "\n";
#endif
  par_reduce_nd_impl<nameT, 1, clover::Range1d, LocalType, LocalAllocator, Empty, Functor, BinaryOp, Finaliser>(
      q, range, result, [](clover::Range1d r) { return r.size; },
      [](sycl::id<1> gid, clover::Range1d r) { return r.from + gid[0]; }, allocator, identity, functor, combiner);
}

template <typename nameT, class LocalType, class LocalAllocator = std::nullptr_t, class Empty = std::nullptr_t,
          class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t, class Finaliser = std::nullptr_t>
static inline void par_reduce_2d(sycl::queue &q, const clover::Range1d &range, sycl::buffer<LocalType, 1> result,
                                 LocalAllocator allocator, const LocalType identity, Functor functor,
                                 BinaryOp combiner) {
#ifdef SYCL_DEBUG
  std::cout << "par_reduce_2d " << range << "\n";
#endif
  par_reduce_nd_impl<nameT, 2, clover::Range2d, LocalType, LocalAllocator, Empty, Functor, BinaryOp, Finaliser>(
      q, range, result, [](clover::Range2d r) { return r.sizeX * r.sizeY; },
      [](sycl::id<1> gid, clover::Range2d r) {
#ifdef SYCL_FLIP_2D
        const size_t x = r.fromX + (gid[0] % (r.sizeY));
        const size_t y = r.fromY + (gid[0] / (r.sizeY));
        return sycl::id<2>(y, x);
#else
        const size_t x = r.fromX + (gid[0] % (r.sizeX));
        const size_t y = r.fromY + (gid[0] / (r.sizeX));
        return sycl::id<2>(x, y);
#endif
      },
      allocator, identity, functor, combiner);
}

} // namespace clover
#endif // CLOVERLEAF_SYCL_SYCL_REDUCTION_HPP
