#include "random.h"

namespace mpi
{
	bool random::_initialized = false;
	std::mt19937 random::_rnd = std::mt19937();
}