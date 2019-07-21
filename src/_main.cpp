#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include "definitions.h"


void test() {
	cl::sycl::queue deviceQueue;

	std::cout << "Device:" << deviceQueue.get_device().get_info<cl::sycl::info::device::vendor>()
	          << std::endl;

	auto xs = Buffer<double, 2>(range<2>(10, 10));

	execute(deviceQueue, [&](cl::sycl::handler &h) {
		cl::sycl::stream os(1024, 128, h);
		auto xsa = xs.access<RW>(h);


		par_ranged<class aa>(h, {0, 0, 10, 10}, [=](id<2> idx) {
			xsa[idx] = 42.3;
		});

		par_ranged<class bb>(h, {0, 0, 10, 10}, [=](id<2> idx) {
			xsa[idx] *= 2;
		});


	});

	auto aa = xs.access<R>();
	std::cout << aa[9][9] << std::endl;


}

int main() {
	test();


	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
