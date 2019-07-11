#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include "definitions.h"

using cl::sycl::accessor;


template<typename T>
class SimpleVadd;


template<typename T, size_t N>
struct Data2 {
	cl::sycl::buffer<T, 2> ab;
	cl::sycl::buffer<T, 1> c;
	Data2(const std::array<T, 2 * N> &VA, std::array<T, N> &VC) :
			ab(VA.data(), cl::sycl::range<2>(2, N)),
			c(VC.data(), cl::sycl::range<1>(N)) {}
};

using namespace std::placeholders;

template<typename T, size_t N>
void doIt(
		const cl::sycl::stream &os,
		cl::sycl::id<1> &wiID,
		const typename Accessor<T, 2, R>::View &accessorAB,
		const typename Accessor<T, 1, W>::View &accessorC) {
	auto i = wiID[0];
	auto b = accessorAB[0][i] + accessorAB[1][i];
	os << i << ", " << wiID[1] << "\n";
	accessorC[wiID] = b * 10;
}

//	std::cout << "Size[" << from << "," << to << "]=" << (to-from) << "\n";




template<typename T, size_t N>
void simple_vadd(const std::array<T, 2 * N> &VA,
                 std::array<T, N> &VC) {
	cl::sycl::queue deviceQueue;
	cl::sycl::range<1> numOfItems{N};


//	Data2<T, N> data(VA, VC);



	auto cb = cl::sycl::buffer<T, 1>(VC.data(), cl::sycl::range<1>(N));
	auto abb = cl::sycl::buffer<T, 2>(VA.data(), cl::sycl::range<2>(2, N));

	Buffer<T, 1> c = Buffer<T, 1>(cb);
	Buffer<T, 2> ab = Buffer<T, 2>(abb);


	deviceQueue.submit([&](cl::sycl::handler &cgh) {


		typename Accessor<T, 1, W>::View accessorC = c.template access<W>(cgh);
		typename Accessor<T, 2, R>::View accessorAB = ab.template access<R>(cgh);


		cl::sycl::stream os(1024, 128, cgh);

		auto kern = [=](cl::sycl::id<1> wiID) {

			auto that = wiID - cl::sycl::id<1>(0);

			accessorC[that] = -1;
//			doIt<T, N>(os, that, accessorAB, accessorC);
			auto i = that[0];
			auto b = accessorAB[0][i] + accessorAB[1][i];
			os << i << ", " << that[1] << "\n";
////			auto a = accessorAB[0][that.];
			accessorC[wiID] = accessorAB[0][i];
//			accessorC[that] = accessorAB[that] + accessorAB[that] + 1;
		};

//		par_ranged1d<class SimpleVadd<T>>(cgh, 2, N - 2,
//		                                  [=](cl::sycl::item<1> a) {
//
//			                                  auto it = a.get_id();
//
//
//			                                  os << a.get_linear_id() << " ->" << it[0] << ","
//			                                     << it[1] << "," << it[2] << "";
//
//			                                  accessorC[a] = it[0];
//
//		                                  });


//		cgh.parallel_for<class AAA>(
//				cl::sycl::range<1>(N - 2), // global range
//				cl::sycl::id<1>(1), // offset
//				[=](cl::sycl::item<1> a) {
//
//					auto it = a.get_id();
//
//
//					os << a.get_linear_id() << " ->" << it[0] << "," << it[1] << "," << it[2] << "";
////					   << "\n";
//
//
//					accessorC[a] = it[0];
//
//				});



//		cgh.parallel_for<class SimpleVadd<T>>(
//				cl::sycl::range<1>(N-2),
//				cl::sycl::id<1>(15),
//				kern);


//		cgh.parallel_for<class SimpleVadd<T>>(cl::sycl::range<1>(N),
//		                                      std::bind(doIt<T, N>,
//		                                                std::cref(os),
//		                                                _1,
//		                                                std::cref(accessorAB),
//		                                                std::cref(accessorC))
//		);
	});
}


void bad() {
	cl::sycl::queue deviceQueue;

	std::cout << "Device:" << deviceQueue.get_device().get_info<cl::sycl::info::device::vendor>()
	          << std::endl;

	deviceQueue.submit([&](cl::sycl::handler &cgh) {

		cl::sycl::stream os(1024, 128, cgh);
		int x_max = 20;
		int y_max = 20;

		int x_min = 10;
		int y_min = 10;


		cgh.parallel_for<class OffsetTest>(
				cl::sycl::range<3>(3, 3, 3), // global range
				cl::sycl::id<3>(0, 0, 0), // offset
				[=](cl::sycl::item<3> a) {
					auto id = a.get_id();
					os << a.get_linear_id() << " ->" << id[0] << "," << id[1] << "," << id[2]
					   << "\n";
				});

	});
	deviceQueue.wait_and_throw();
}

int main() {
	std::array<cl::sycl::cl_int, 12> A = {
			{
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12
			}
	};
	std::array<cl::sycl::cl_int, 6> Out = {};
//
	simple_vadd(A, Out);

//	bad();

	for (const auto &x : Out) {
		std::cout << "=" << x << std::endl;
	}


	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
