#pragma once
#include <iostream>
#include "mpi.h"

namespace mpi {
	template<typename F, F f>
	struct factor {
		static constexpr F value = f;
		typedef F factor_type;
		typedef factor type;
		constexpr operator factor_type()   const noexcept { return value; }
		constexpr factor_type operator()() const noexcept { return value; }
	};

	struct seconds : factor<size_t, 1> 
		{ static constexpr auto name = "seconds"; };
	struct milliseconds : factor<size_t, 1000> 
		{ static constexpr auto name = "milliseconds"; };
	struct microseconds : factor<size_t, 1000000>
		{ static constexpr auto name = "microseconds"; };

	template<typename Unit = seconds>
	struct mpi_clock {
		typedef size_t time_point;
		static time_point now() 
			{ return static_cast<time_point>(MPI_Wtime() * Unit::value); }
	};

	///<summary>
	/// Класс для замера времени
	/// выполнения программы 
	///</summary>
	template<typename Unit> class mpi_timer {
	private:
		typename mpi_clock<Unit>::time_point _start, _end;
		int _rank;
		int _currentRank;

	private:
		///<summary>
		/// Начало отчета времени для указанного процесса
		///</summary>
		void start() {
			if (_currentRank == _rank)
				_start = mpi::mpi_clock<Unit>::now();
		}

		///<summary>
		/// Завершение отчета времени
		///</summary>
		void end() {
			if (_currentRank == _rank)
				_end = mpi::mpi_clock<Unit>::now();
		}

		///<summary>
		/// Вывод результата
		///</summary>
		void print() const {
			if (_currentRank == _rank)
				std::cout << "[Timer] Operation took " << _end - _start << " " 
						  << Unit::name << std::endl;
		}
	public:
		///<summary>
		/// Инициализация таймера и запуск начала отчета
		///</summary>
		explicit mpi_timer(int rank) {
			_rank = rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &_currentRank);
			start();
		}

		///<summary>
		/// Заканчивает отчет времени и выводит
		/// результата на экран
		///</summary>
		~mpi_timer() {
			end(); 
			print();
		}
	};
}