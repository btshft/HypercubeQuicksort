#pragma once
#include <random>
#include <algorithm>

namespace mpi
{
	///<summary>
	/// —генерировать рандомное что-то
	///</summary>
	class random
	{
		// ...
		typedef int seed_t;
		typedef std::mt19937 engine;
		typedef std::uniform_int_distribution<seed_t> uniform_seed_distr;
		typedef std::numeric_limits<int> int_limits;

		// ћолодой человек, ¬ы что не видите,
		// что у нас тут статический класс
		// не надо его присваивать!
		public:
			random()         = delete;
			random(random&)  = delete;
			random(random&&) = delete;
			random & operator=(const random&) = delete;

		private:
			static engine _rnd;
			static bool _initialized;
			static void init() {
				if (_initialized) return;
				_rnd.seed(std::random_device()());
			}

		public:
			///<summary>
			/// —генерировать рандомное целое число
			///</summary>
			static int integer(int from = int_limits::min(), int to = int_limits::max()) {
				init();
				return uniform_seed_distr(from, to)(_rnd);
			}

			///<summary>
			/// —генерировать массив случайных чисел
			///</summary>
			static std::vector<int> integers(int count, 
				int from = int_limits::min(), int to = int_limits::max()) {
				init();
				std::vector<int> data(count);
				std::generate(std::begin(data), std::end(data), 
					[from, to]{ return integer(from, to); }
				);
				return data;
			}

			///<summary>
			/// —генерировать массив случайных чисел по двум итераторам
			///</summary>
			template<typename It>
			static void generate(It range_from, It range_to, 
				int from = int_limits::min(), int to = int_limits::max())
			{
				init();
				std::generate(range_from, range_to,
					[from, to] { return integer(from, to); }
				);
			}
	};
}
