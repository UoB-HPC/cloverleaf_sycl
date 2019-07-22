#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include "definitions.h"



//template<class T,
//		int kDims,
//		cl::sycl::access::target kTarget,
//		class BinaryOperation>
//struct reduction_info_impl {
//	accessor<T, kDims, RW, kTarget> &var;
//	const T &identity;
//	BinaryOperation combiner;
//	reduction_info_impl(
//			accessor<T, kDims, RW, kTarget> &var,
//			const T &identity,
//			BinaryOperation combiner) : var(var), identity(identity), combiner(combiner) {}
//};
//
//template<class T, int kDims,
//		cl::sycl::access::target kTarget,
//		class BinaryOperation>
//reduction_info_impl<T, kDims, kTarget, BinaryOperation>
//reduction(accessor<T, kDims, RW, kTarget> &var, BinaryOperation combiner) {
//	return reduction_info_impl<T, kDims, kTarget, BinaryOperation>(var, 0, combiner);
//}
//
//template<class T, int kDims,
//		cl::sycl::access::target kTarget,
//		class BinaryOperation>
//reduction_info_impl<T, kDims, kTarget, BinaryOperation>
//reduction(accessor<T, kDims, RW, kTarget> &var, const T &identity, BinaryOperation combiner) {
//	return reduction_info_impl<T, kDims, kTarget, BinaryOperation>(var, identity, combiner);
//}
//
//template<typename nameT, typename functorT, int dimensions>
//inline void
//parallel_for(cl::sycl::handler &cgh, range<dimensions> range, reduction_info_impl<>, const functorT &functor) {
//	if (DEBUG)
//		std::cout << "par_ranged 1d:x=" << range.from << "(" << range.size << ")" << std::endl;
//	cgh.parallel_for<nameT>(
//			cl::sycl::range<1>(range.size),
//			cl::sycl::id<1>(range.from),
//			functor);
//}


size_t next_p2(size_t a) {
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


template<typename nameT,
		class LocalAllocator, class Empty, class Functor, class BinaryOp, class Finaliser>
void par_reduce(cl::sycl::queue &q, const Range1d &range,
                LocalAllocator allocator,
                Empty empty,
                Functor functor,
                BinaryOp combiner,
                Finaliser finaliser) {

	size_t maxWgSize = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

//	std::cout << "N=" << range.size << " Mwg=" << maxWgSize << "\n";
	size_t length = range.size;

	do {
//			std::cout << "\tLL=" << length << "\n";
		auto paddedN = next_p2(length);
		auto wgN = std::min(paddedN, maxWgSize);

		q.submit([=](cl::sycl::handler &h) mutable {
			auto scratch = allocator(h, paddedN);
//				std::cout << "\tL2=" << paddedN << " wg=" << wgN << "\n";
			h.parallel_for<nameT>(
					cl::sycl::nd_range<1>(cl::sycl::range<1>(paddedN),
					                      cl::sycl::range<1>(wgN),
					                      id<1>(range.from)),
					[=](cl::sycl::nd_item<1> id) {
						cl::sycl::id<1> globalid = id.get_global_id();
						cl::sycl::id<1> localid = id.get_local_id();

						if (globalid[0] >= length) empty(scratch, localid);
						else functor(scratch, localid, globalid);

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
		q.wait_and_throw();
		length = length / wgN;
	} while (length > 1);
}


template<typename T, typename U, typename C>
struct local_reduction {

	cl::sycl::accessor<T, 1,
			cl::sycl::access::mode::read_write,
			cl::sycl::access::target::local> local;

	C actual;
	cl::sycl::accessor<U, 1, RW, cl::sycl::access::target::global_buffer> result;

	local_reduction(handler &h, size_t size, C actual, cl::sycl::buffer<U, 1> &b) :
			local(range<1>(size), h),
			actual(actual),
			result(b.template get_access<RW>(h)) {}

//	local_reduction(local_reduction<T, U, kMode, kTarget>  &b) : scratch(b.scratch), actual(b.actual) {}

};

typedef local_reduction<int, int, accessor<int, 1, RW, cl::sycl::access::target::global_buffer >> my_reduction;

void test(int in) {
	cl::sycl::queue deviceQueue;

//	std::cout << "Device:" << deviceQueue.get_device().get_info<cl::sycl::info::device::vendor>()
//	          << std::endl;

	int N = in;

	auto xs = Buffer<int, 1>(range<1>(N));
	auto ys = Buffer<int, 1>(range<1>(N));

	execute(deviceQueue, [&](handler &g) {
		auto view = ys.access<RW>(g);
		par_ranged<class foo>(g, {0, N}, [=](id<1> idx) {
			view[idx] = idx[0];
		});

	});


//	auto sum = Buffer<int, 1>(range<1>(WG));

//	execute(deviceQueue, [&](cl::sycl::handler &h) {
//		cl::sycl::stream os(1024, 128, h);
//	auto xsa = xs.access<RW>(h);
////		auto localSums = ys.access<RW>(h);
//		auto sumA = sum.access<RW>(h);


//		par_ranged<class aa>(h, {0, 0, 10, 10}, [=](id<2> idx) {
//			xsa[idx] = 42.3;
//		});



	par_reduce<class bb>(deviceQueue, {0, N},
	                     [=](handler &h, size_t &size) mutable {
		                     return my_reduction(h, size, ys.access<RW>(h), xs.buffer);
	                     },
	                     [](my_reduction ctx, id<1> lidx) { ctx.local[lidx] = 0; },
	                     [](my_reduction ctx, id<1> lidx, id<1> gidx) { ctx.local[lidx] = (ctx.actual[gidx]); },
	                     [](my_reduction ctx, id<1> idx, id<1> idy) { ctx.local[idx] += ctx.local[idy]; },
	                     [](my_reduction ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });


//	});

	auto ysaa = xs.access<R>();
//	for (int i = 0; i < N; ++i) {
//		std::cout << "[" << i << "] = " << ysaa[i] << std::endl;
//	}

	auto expected = 0;
	for (int i = 0; i < in; ++i) {
		expected += i;
	}

	if (ysaa[0] != expected) {
		std::cout << in << " : " << expected << "!=" << ysaa[0] << "=FAIL\n";
	} else {
		std::cout << in << " : " << expected << "=OK\n";
	}

//	auto aa = sum.access<R>();
//	std::cout << "->" << aa[0] << " " << aa[1] << std::endl;




//	auto aa = xs.access<R>();
//	std::cout << aa[9][9] << std::endl;
	deviceQueue.wait_and_throw();

}

int main() {
	for (int i = 2; i < 128; ++i) {
		test(i);

	}


	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
