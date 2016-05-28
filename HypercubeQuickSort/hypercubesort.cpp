
#include <exception>
#include "parallel.h"
#include "pretty.hpp"
#include <iostream>
#include "timer.h"
#include "random.h"

using namespace mpi;

using std::cout;
using std::endl;

void main(int argc, char** argv)
{
	mpi::init(&argc, &argv);
	auto rank = mpi::getRank(MPI_COMM_WORLD);
	auto size = mpi::getSize(MPI_COMM_WORLD);

	mpi::shared_array<int> data(100000);
	if (rank == 0) {
		mpi::random::generate(std::begin(data), std::end(data), -1000, 1000);
		std::cout << "[10 START] Original data: " << std::vector<int>(std::begin(data), std::begin(data) + 10) << std::endl;
		std::cout << "[10 END] Original data: " << std::vector<int>(std::end(data) - 10, std::end(data)) << std::endl;
		std::cout << "\n[ROOT] Dataset size: " << data.size() << std::endl;
		if (size > 1)
			std::cout << "\nStarting parallel sort with " << size << " processes\n";
		else
			std::cout << "\nStarting sequential sort (std::sort)\n";
	}

	if (size > 1) {
		with(mpi::mpi_timer<microseconds> timer(0))
			mpi::sorter<int>::sort(data);
	} else {
		with(mpi_timer<microseconds> timer(0))
			std::sort(std::begin(data), std::end(data));
	}

	if (rank == 0) {
		std::cout << "[10 START]Sorted data: " << std::vector<int>(std::begin(data), std::begin(data) + 10) << std::endl;
		std::cout << "[10 END] Sorted data: " << std::vector<int>(std::end(data) - 10, std::end(data)) << std::endl;
		std::cout << "\n[ROOT] Sorted dataset size: " << data.size() << std::endl;
	}

	mpi::finalize();
}