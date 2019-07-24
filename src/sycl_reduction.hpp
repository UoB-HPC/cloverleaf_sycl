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


static inline size_t next_powerof2(size_t a) {
	auto v = a;
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
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

};


template<typename nameT,
		size_t dimension,
		class RangeTpe,
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

	//range.sizeX * range.sizeY

	size_t length = lengthFn(range);
	do {
		size_t paddedN = next_powerof2(length);
		const size_t wgN = std::min(paddedN, maxWgSize);

		q.submit([=](cl::sycl::handler &h) mutable {
			auto scratch = allocator(h, paddedN);
			h.parallel_for<nameT>(
					cl::sycl::nd_range<1>(cl::sycl::range<1>(paddedN),
					                      cl::sycl::range<1>(wgN)),
					[=](cl::sycl::nd_item<1> id) {
						cl::sycl::id<1> globalid = id.get_global_id();
						cl::sycl::id<1> localid = id.get_local_id();

						if (globalid[0] >= length) empty(scratch, localid);
						else {


//							const size_t x = globalid[0] % range.sizeX + range.fromX;
//							const size_t y = globalid[0] / range.sizeY + range.fromY;
//							cl::sycl::id<2>(x, y)

							functor(scratch, localid, rangeIdFn(globalid[0], range));
						}

						id.barrier(cl::sycl::access::fence_space::local_space);
						if (globalid[0] < paddedN) {
							for (size_t offset = paddedN / 2; offset > 0; offset >>= 1u) {
								if (localid[0] < offset) combiner(scratch, localid, localid + offset);
								id.barrier(cl::sycl::access::fence_space::local_space);
							}
							if (localid[0] == 0) finaliser(scratch, id.get_group(0), localid);
						}
					});
		});
		length = length / wgN;
	} while (length > 1);
	q.wait();
}


template<typename nameT,
		class LocalAllocator = std::nullptr_t,
		class Empty = std::nullptr_t,
		class Functor = std::nullptr_t,
		class BinaryOp = std::nullptr_t,
		class Finaliser = std::nullptr_t>
static inline void par_reduce_2d(cl::sycl::queue &q, const Range2d &range,
                                 LocalAllocator allocator,
                                 Empty empty,
                                 Functor functor,
                                 BinaryOp combiner,
                                 Finaliser finaliser) {
	par_reduce_nd_impl<nameT, 2, Range2d,
			LocalAllocator,
			Empty,
			Functor,
			BinaryOp,
			Finaliser
	>(q, range,
	  [](Range2d r) { return r.sizeX * r.sizeY; },
	  [](cl::sycl::id<1> gid, Range2d r) {
		  const size_t x = r.fromX + (gid[0] % (r.sizeX));
		  const size_t y = r.fromY + (gid[0] / (r.sizeX));
		  return cl::sycl::id<2>(x, y);
	  },
	  allocator, empty, functor, combiner, finaliser);
}

template<typename nameT,
		class LocalAllocator = std::nullptr_t,
		class Empty = std::nullptr_t,
		class Functor = std::nullptr_t,
		class BinaryOp = std::nullptr_t,
		class Finaliser = std::nullptr_t>
static inline void par_reduce_1d(cl::sycl::queue &q, const Range1d &range,
                                 LocalAllocator allocator,
                                 Empty empty,
                                 Functor functor,
                                 BinaryOp combiner,
                                 Finaliser finaliser) {
	par_reduce_nd_impl<nameT, 1, Range1d,
			LocalAllocator,
			Empty,
			Functor,
			BinaryOp,
			Finaliser
	>(q, range,
	  [](Range1d r) { return r.size; },
	  [](cl::sycl::id<1> gid, Range1d r) { return r.from + gid; },
	  allocator, empty, functor, combiner, finaliser);
}

//
//template<typename nameT,
//		class LocalAllocator = std::nullptr_t,
//		class Empty= std::nullptr_t,
//		class Functor= std::nullptr_t,
//		class BinaryOp= std::nullptr_t,
//		class Finaliser= std::nullptr_t>
//void par_reduce(cl::sycl::queue &q, const Range1d &range,
//                LocalAllocator allocator,
//                Empty empty,
//                Functor functor,
//                BinaryOp combiner,
//                Finaliser finaliser) {
//	const size_t maxWgSize = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
//	size_t length = range.size;
//	do {
//		const size_t paddedN = next_powerof2(length);
//		const size_t wgN = std::min(paddedN, maxWgSize);
//
//		q.submit([=](cl::sycl::handler &h) mutable {
//			auto scratch = allocator(h, paddedN);
//			h.parallel_for<nameT>(
//					cl::sycl::nd_range<1>(cl::sycl::range<1>(paddedN),
//					                      cl::sycl::range<1>(wgN),
//					                      cl::sycl::id<1>(range.from)),
//					[=](cl::sycl::nd_item<1> id) {
//						cl::sycl::id<1> globalid = id.get_global_id();
//						cl::sycl::id<1> localid = id.get_local_id();
//
//						if (globalid[0] >= length) empty(scratch, localid);
//						else functor(scratch, localid, globalid);
//
//						id.barrier(cl::sycl::access::fence_space::local_space);
//						if (globalid[0] < paddedN) {
//							for (size_t offset = paddedN / 2; offset > 0; offset >>= 1u) {
//								if (localid[0] < offset) combiner(scratch, localid, localid + offset);
//								id.barrier(cl::sycl::access::fence_space::local_space);
//							}
//							if (localid[0] == 0) finaliser(scratch, id.get_group(0), localid);
//						}
//					});
//		});
//		length = length / wgN;
//	} while (length > 1);
//	q.wait();
//}

#endif //CLOVERLEAF_SYCL_SYCL_REDUCTION_HPP
