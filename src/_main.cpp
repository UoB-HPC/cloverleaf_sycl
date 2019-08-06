#include "definitions.h"
#include <array>
#include <iostream>


class foo;

void test(int in) {
	cl::sycl::queue deviceQueue;

	std::cout << "Device:" << deviceQueue.get_device().get_info<cl::sycl::info::device::vendor>()
	          << std::endl;

	int N = in;

	auto xs = clover::Buffer<int, 1>(range<1>(N));
	auto ys = clover::Buffer<int, 1>(range<1>(N));

	clover::execute(deviceQueue, [&](handler &g) {
		auto view = ys.access<RW>(g);

		g.parallel_for<foo>(
				cl::sycl::range<1>(N),
				cl::sycl::id<1>(0),
				[=](id<1> idx) {
					view[idx] = idx[0];
				});


	});


	auto ysaa = xs.access<R>();

	auto expected = 0;
	for (int i = 0; i < in; ++i) {
		expected += i;
	}

	if (ysaa[0] != expected) {
		std::cout << in << " : " << expected << "!=" << ysaa[0] << "=FAIL\n";
	} else {
		std::cout << in << " : " << expected << "=OK\n";
	}

	deviceQueue.wait_and_throw();

}

int main() {
	for (int i = 2; i < 128; ++i) {
		test(i);
	}


	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
