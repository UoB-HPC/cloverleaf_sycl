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

#include <CL/sycl.hpp>
#include <iostream>
#include <utility>
#include "sycl_utils.hpp"

namespace clover {
	static inline size_t next_powerof2(size_t x) {
		x--;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x + 1;
	}


	static inline size_t prev_powerof2(size_t x) {
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x - (x >> 1);
	}


	template<typename T, typename U, typename C>
	struct local_reducer {

		cl::sycl::accessor<T, 1,
				cl::sycl::access::mode::read_write,
				cl::sycl::access::target::local> local;

		C actual;
		cl::sycl::accessor<U, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> result;

		local_reducer(cl::sycl::handler &h, size_t size, C actual, cl::sycl::buffer<U, 1> &b) :
				local(cl::sycl::range<1>(size), h),
				actual(actual),
				result(b.template get_access<cl::sycl::access::mode::read_write>(h)) {}

		inline void drain(cl::sycl::id<1> lid, cl::sycl::id<1> gid) const { local[lid] = result[gid]; }

	};


	template<typename nameT,
			size_t dimension,
			class RangeTpe,
			class LocalType,
			class LocalAllocator = std::nullptr_t,
			class Empty= std::nullptr_t,
			class Functor= std::nullptr_t,
			class BinaryOp= std::nullptr_t,
			class Finaliser= std::nullptr_t,
			class RangeLengthFn = std::nullptr_t,
			class RangeIdFn = std::nullptr_t>
	static inline void par_reduce_nd_impl(cl::sycl::queue &q,
	                                      RangeTpe range, RangeLengthFn lengthFn, RangeIdFn rangeIdFn,
	                                      LocalAllocator allocator,
	                                      Empty empty,
	                                      Functor functor,
	                                      BinaryOp combiner,
	                                      Finaliser finaliser) {

		const size_t maxWgSize = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
		const size_t localMemSize = q.get_device().get_info<cl::sycl::info::device::local_mem_size>();

		size_t unpadded = lengthFn(range);
		size_t length = next_powerof2(unpadded);

		size_t local = std::min(length, maxWgSize);
		if (local * sizeof(LocalType) > localMemSize)
			local = prev_powerof2(localMemSize / sizeof(LocalType));

		bool functorDone = false;
		do {

			if (SYCL_DEBUG) std::cout << "RD: Local=" << local << " Len=" << length << " unpadded:" << unpadded << "\n";

			q.submit([=](cl::sycl::handler &h) mutable {
				auto scratch = allocator(h, local);
				size_t min = (length < local) ? length : local;
				h.parallel_for<nameT>(
						cl::sycl::nd_range<1>(cl::sycl::range<1>(std::max(local, length)),
						                      cl::sycl::range<1>(std::min(local, length))),
						[=](cl::sycl::nd_item<1> id) {
							const cl::sycl::id<1> globalid = id.get_global_id();
							const cl::sycl::id<1> localid = id.get_local_id();

							if (globalid[0] >= unpadded) empty(scratch, localid);
							else {
								if (functorDone) scratch.drain(localid, globalid[0]);
								else functor(scratch, localid, rangeIdFn(globalid[0], range));
							}

							id.barrier(cl::sycl::access::fence_space::local_space);

							if (globalid[0] < length) {
								for (size_t offset = min / 2; offset > 0; offset >>= 1) {
									if (localid[0] < offset) combiner(scratch, localid, localid + offset);
									id.barrier(cl::sycl::access::fence_space::local_space);
								}
								if (localid[0] == 0) finaliser(scratch, id.get_group(0), localid);
							}
						});
			});
			if (SYCL_DEBUG) q.wait_and_throw();
			length = length / local;
			functorDone = true;
		} while (length > 1);
		if (SYCL_DEBUG){
			q.wait_and_throw();
			if (SYCL_DEBUG) std::cout << "RD: done= " << length << "\n";
		}
	}


	template<typename nameT,
			class LocalType,
			class LocalAllocator = std::nullptr_t,
			class Empty = std::nullptr_t,
			class Functor = std::nullptr_t,
			class BinaryOp = std::nullptr_t,
			class Finaliser = std::nullptr_t>
	static inline void par_reduce_2d(cl::sycl::queue &q, const clover::Range2d &range,
	                                 LocalAllocator allocator,
	                                 Empty empty,
	                                 Functor functor,
	                                 BinaryOp combiner,
	                                 Finaliser finaliser) {
		if (SYCL_DEBUG) std::cout << "PR2d " << range << "\n";
		par_reduce_nd_impl<nameT, 2, clover::Range2d,
				LocalType,
				LocalAllocator,
				Empty,
				Functor,
				BinaryOp,
				Finaliser
		>(q, range,
		  [](clover::Range2d r) { return r.sizeX * r.sizeY; },
		  [](cl::sycl::id<1> gid, clover::Range2d r) {
			  const size_t x = r.fromX + (gid[0] % (r.sizeX));
			  const size_t y = r.fromY + (gid[0] / (r.sizeX));
#ifdef SYCL_FLIP_2D
			  return cl::sycl::id<2>(y, x);
#else
			  return cl::sycl::id<2>(x, y);
#endif
		  },
		  allocator, empty, functor, combiner, finaliser);
	}

	template<typename nameT,
			class LocalType,
			class LocalAllocator = std::nullptr_t,
			class Empty = std::nullptr_t,
			class Functor = std::nullptr_t,
			class BinaryOp = std::nullptr_t,
			class Finaliser = std::nullptr_t>
	static inline void par_reduce_1d(cl::sycl::queue &q, const clover::Range1d &range,
	                                 LocalAllocator allocator,
	                                 Empty empty,
	                                 Functor functor,
	                                 BinaryOp combiner,
	                                 Finaliser finaliser) {
		if (SYCL_DEBUG) std::cout << "PR1d " << range << "\n";
		par_reduce_nd_impl<nameT, 1, clover::Range1d,
				LocalType,
				LocalAllocator,
				Empty,
				Functor,
				BinaryOp,
				Finaliser
		>(q, range,
		  [](clover::Range1d r) { return r.size; },
		  [](cl::sycl::id<1> gid, clover::Range1d r) { return r.from + gid; },
		  allocator, empty, functor, combiner, finaliser);
	}

}


#endif //CLOVERLEAF_SYCL_SYCL_REDUCTION_HPP
