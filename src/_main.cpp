#include "definitions.h"
#include "sycl_reduction.hpp"
#include <array>
#include <iostream>
#include <sstream>


class foo;

void print_device(const cl::sycl::device &device) {
	auto exts = device.get_info<cl::sycl::info::device::extensions>();
	std::ostringstream extensions;
	std::copy(exts.begin(), exts.end(), std::ostream_iterator<std::string>(extensions, ","));

	auto type = device.get_info<cl::sycl::info::device::device_type>();
	auto typeName = "(unknown)";
	//@formatter:off
	switch (type){
		case sycl::info::device_type::cpu: typeName = "CPU"; break;
		case sycl::info::device_type::gpu: typeName = "GPU"; break;
		case sycl::info::device_type::accelerator: typeName = "ACCELERATOR"; break;
		case sycl::info::device_type::custom: typeName = "CUSTOM"; break;
		case sycl::info::device_type::automatic: typeName = "AUTOMATIC"; break;
		case sycl::info::device_type::host: typeName = "HOST"; break;
		case sycl::info::device_type::all: typeName = "ALL"; break;
	}
	//@formatter:on
	cl::sycl::platform platform = device.get_platform();
	std::cout << "[SYCL] Device        : " << device.get_info<cl::sycl::info::device::name>() << "\n";
	std::cout << "[SYCL]  - Type       : " << typeName << "\n";
	std::cout << "[SYCL]  - Vendor     : " << device.get_info<cl::sycl::info::device::vendor>() << "\n";
	std::cout << "[SYCL]  - Extensions : " << extensions.str() << "\n";
	std::cout << "[SYCL]  - Platform   : " << platform.get_info<cl::sycl::info::platform::name>() << "\n";
	std::cout << "[SYCL]     - Vendor  : " << platform.get_info<cl::sycl::info::platform::vendor>() << "\n";
	std::cout << "[SYCL]     - Version : " << platform.get_info<cl::sycl::info::platform::version>() << "\n";
	std::cout << "[SYCL]     - Profile : " << platform.get_info<cl::sycl::info::platform::profile>() << "\n";
}


void test(cl::sycl::queue deviceQueue, const size_t N) {


	std::vector<double> data(N);

	double expected = 1.0e+18;
	for (size_t i = 0; i < N; ++i) {
		auto x = static_cast<double>(i);
		expected = std::min(expected, 1000 + x);
		data[i] = 1000 + x;
	}

	auto xs = clover::Buffer<double, 1>(data.begin(), data.end());


	struct captures {
		clover::Accessor<double, 1, R>::Type data;
	};
	typedef clover::local_reducer<double, double, captures> ctx;
	clover::Buffer<double, 1> result((range<1>(N)));

	clover::par_reduce_1d<class field_summary, double>(
			deviceQueue, clover::Range1d(0, N),
			[=](handler &h, size_t &size) mutable {
				return ctx(h, size, {xs.access<R>(h)}, result.buffer);
			},
			[](const ctx &ctx, id<1> lidx) { ctx.local[lidx] = 1.0e+18; },
			[](ctx ctx, id<1> lidx, id<1> idx) {
				ctx.local[lidx] = cl::sycl::fmin(ctx.local[lidx], ctx.actual.data[idx]) ;
//				ctx.local[lidx] += ctx.actual.data[idx];
			},
			[](const ctx &ctx, id<1> idx, id<1> idy) {
				ctx.local[idx] = cl::sycl::fmin(ctx.local[idx], ctx.local[idy]) ;
//				ctx.local[idx] += ctx.local[idy];
			},
			[](const ctx &ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });


	double actual = result.access<R>()[0];


	if (actual != expected) {
		std::cout << N << " : " << expected << "!=" << actual << "=FAIL\n";
	} else {
		std::cout << N << " : " << expected << "=OK\n";
	}

	deviceQueue.wait_and_throw();

}

int main() {
	int start = 14745600;
	int range = 10;
	cl::sycl::queue deviceQueue;

	print_device(deviceQueue.get_device());

	for (int i = start; i < start + range; ++i) {
		test(deviceQueue, i);
	}

	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
