#pragma once

#include <type_traits>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <iostream>
#include "shared_array.h"

#define MPI_THROW(message, comm)         \
		do {                             \
			std::cerr << "MPIEXT FATAL ERROR: " << message << std::endl;     \
			MPI_Abort(comm, -1);         \
		} while(0)                       \


namespace mpi {

	// Некоторые трейты 
	// для ограничения шаблонного параметра
	namespace traits { 
		template<typename T> 
		struct is_vector: std::false_type {};
		template<typename T, typename A>
		struct is_vector<std::vector<T, A>>: std::true_type {};
		template <typename T>
		struct identity { typedef T type; };
		template<typename T>
		struct is_shared_array : std::false_type {};
		template<typename T>
		struct is_shared_array<mpi::shared_array<T>> : std::true_type {};
	}

	#define GET_VALUE_TYPE(T)        typename T::value_type
	#define ENABLE_IF_FUNDAMENTAL(T) typename std::enable_if<std::is_fundamental<T>::value, int>::type* = nullptr
	#define ENABLE_IF_CLASS(T)  typename std::enable_if<std::is_class<T>::value, int>::type* = nullptr
	#define ENABLE_IF_VECTOR(T) typename std::enable_if<mpi::traits::is_vector<T>::value, int>::type* = nullptr
	#define ENABLE_IF_SARRAY(T) typename std::enable_if<mpi::traits::is_shared_array<T>::value, int>::type* = nullptr

	// Определяет тип 
	template<typename T>
	MPI_Datatype get_mpi_datatype()
	{
		static_assert(std::is_fundamental<T>::value, "Error");
		if (std::is_same<T, int>::value) {
			return MPI_INTEGER;
		}
		if (std::is_same<T, short>::value) {
			return MPI_SHORT_INT;
		}
		if (std::is_same<T, float>::value) {
			return MPI_FLOAT;
		}
		if (std::is_same<T, double>::value) {
			return MPI_DOUBLE;
		}
		if (std::is_same<T, char>::value) {
			return MPI_CHAR;
		}
		if (std::is_same<T, bool>::value) {
			return MPI_BYTE;
		}
		if (std::is_same<T, long>::value) {
			return MPI_LONG;
		}
		return MPI_DATATYPE_NULL;
	}

	// MPI_Init alias
	inline void init(int* argc, char*** argv)
	{
		MPI_Init(argc, argv);
	}

	// MPI_Finalize alias
	inline void finalize()
	{
		MPI_Finalize();
	}

	// MPI_Barrier alias
	inline void barrier(MPI_Comm comm = MPI_COMM_WORLD)
	{
		MPI_Barrier(comm);
	}

	// MPI_Comm_size alias
	inline int getSize(MPI_Comm comm)
	{
		int _size;
		MPI_Comm_size(comm, &_size);
		return _size;
	}

	// MPI_Comm_rank alias
	inline int getRank(MPI_Comm comm)
	{
		int _rank;
		MPI_Comm_rank(comm, &_rank);
		return _rank;
	}

	// Отправляет базовый тип указанному получателю
	template<typename T, ENABLE_IF_FUNDAMENTAL(T)>
	void send(const T what, int dest, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
		auto type = get_mpi_datatype<T>();
		MPI_Send(&what, 1, type, dest, tag, comm);
	}

	// Принимает базовый тип от указанного отправителя
	template<typename T, ENABLE_IF_FUNDAMENTAL(T)>
	T receive(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD) {
		T what;
		auto type = get_mpi_datatype<T>();
		MPI_Recv(&what, 1, type, source, tag, comm, MPI_STATUS_IGNORE);
		return what;
	}

	// Отправляет вектор указаному получателю
	template<typename T, ENABLE_IF_VECTOR(T)>
	void send(const T& what, int dest, int tag, MPI_Comm comm = MPI_COMM_WORLD)
	{
		auto type = get_mpi_datatype<typename T::value_type>();
		int len = what.size();
		MPI_Send(&len, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
		if (len > 0)
			MPI_Send(&what[0], len, type, dest, tag, comm);
	}

	// Принимает вектор от указанного отправителя
	template<typename T, ENABLE_IF_VECTOR(T)>
	T receive(int source, int tag, MPI_Comm comm = MPI_COMM_WORLD)
	{
		int len; 
		MPI_Recv(&len, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
		if (len < 1)
			return T{};
		T vec(len);
		auto type = get_mpi_datatype<typename T::value_type>();
		MPI_Recv(&vec[0], len, type, source, tag, comm, MPI_STATUS_IGNORE);
		return vec;		
	}

	// Операция отправки и приема в одной
	template<typename T, ENABLE_IF_VECTOR(T)>
	T sendreceive(const T& what, int dest, int source, int tag, MPI_Comm comm = MPI_COMM_WORLD)
	{
		int newLen = 0, oldLen = what.size();
		MPI_Sendrecv(&oldLen, 1, MPI_INT, dest, tag, &newLen, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
		if (newLen < 1)
			return T{};
		T newVec(newLen);
		auto type = get_mpi_datatype<typename T::value_type>();
		MPI_Sendrecv(&what[0], oldLen, type, dest, tag, &newVec[0], newLen, type, source, tag, comm, MPI_STATUS_IGNORE);
		return newVec;
	}

	//...
	template<typename T, ENABLE_IF_SARRAY(T)>
	T sendreceive(const T& what, int dest, int source, int tag, MPI_Comm comm = MPI_COMM_WORLD)
	{
		int newLen = 0, oldLen = what.size();
		MPI_Sendrecv(&oldLen, 1, MPI_INT, dest, tag, &newLen, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
		//if (newLen < 1 || oldLen < 1)
		//	return T{};
		T newArr(newLen);
		auto type = get_mpi_datatype<typename T::value_type>();
		MPI_Sendrecv(what.get(), oldLen, type, dest, tag, newArr.get(), newLen, type, source, tag, comm, MPI_STATUS_IGNORE);
		return newArr;
	}

	// Широковещательная операция для базовых типов
	template<typename T, ENABLE_IF_FUNDAMENTAL(T)>
	void broadcast(T* value, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		auto type = get_mpi_datatype<T>();
		MPI_Bcast(value, 1, type, root, comm);
	}

	// Широковещательная операция для векторов
	template<typename T, ENABLE_IF_VECTOR(T)>
	void broadcast(T* value, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		int rank, len;
		MPI_Comm_rank(comm, &rank);
		if (rank == root)
			len = value->size();
		MPI_Bcast(&len, 1, MPI_INT, root, comm);
		if (len > 0) {
			value->resize(len);
			auto type = get_mpi_datatype<typename T::value_type>();
			MPI_Bcast(&(*value)[0], len, type, root, comm);
		}
	}


	// Рассылка по одному элементу базового типа на каждый из процессов
	template<typename T, ENABLE_IF_VECTOR(T)>
	auto scatter(const T& values, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		static_assert(std::is_fundamental<typename T::value_type>::value, "Fundamentals only");
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		//if (rank == root)
		//	len = values.size();
		//MPI_Bcast(&len, 1, MPI_INT, root, comm);
		//values.resize(len);
		typedef typename T::value_type value_type;
		value_type value{};
		auto type = get_mpi_datatype<value_type>();
		MPI_Scatter(&values[0], 1, type, &value, 1, type, root, comm);
		return value;
	}

	// Рассылает i-му процессу необходимое кол-во данных
	template<typename T, ENABLE_IF_VECTOR(T)>
	T scatter(const T& values, const std::vector<int>& counts, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		typedef typename T::value_type inner;
		int rank, size, *displs = nullptr;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		// Проверяем на соответствие кол-ва запрошенных
		// элементов и кол-ва элементо всего
		if (rank == root) {
			int sum = std::accumulate(counts.begin(), counts.end(), 0);
			if (sum > values.size())
				MPI_THROW("Values array has less items than was requested", comm);
		}
		// Вектор для возврата
		T vec{};
		vec.resize(counts[rank]);
		// Считаем смещения в начальном векторе данных
		if (rank == root) {
			displs = new int[size];
			displs[0] = 0;
			for (auto pe = 1; pe < size; pe++)
				displs[pe] = displs[pe - 1] + counts[pe - 1];
		}
		auto type = get_mpi_datatype<inner>();
		MPI_Scatterv(&values[0], &counts[0], displs, type, &vec[0], counts[rank], type, root, comm);
		if (rank == root)
			delete[] displs;
		return vec;
	}

	// ....
	template<typename T, ENABLE_IF_SARRAY(T)>
	T scatter(const T& values, const std::vector<int>& counts, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		typedef typename T::value_type inner;
		int rank, size, *displs = nullptr;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		// Проверяем на соответствие кол-ва запрошенных
		// элементов и кол-ва элементо всего
		if (rank == root) {
			int sum = std::accumulate(counts.begin(), counts.end(), 0);
			if (sum > values.size())
				MPI_THROW("Values array has less items than was requested", comm);
		}
		// Массив для возврата
		T arr{};
		arr.reallocate(counts[rank]);
		// Считаем смещения в начальном векторе данных
		if (rank == root) {
			displs = new int[size];
			displs[0] = 0;
			for (auto pe = 1; pe < size; pe++)
				displs[pe] = displs[pe - 1] + counts[pe - 1];
		}
		auto type = get_mpi_datatype<inner>();
		MPI_Scatterv(values.get(), &counts[0], displs, type, arr.get(), counts[rank], type, root, comm);
		if (rank == root)
			delete[] displs;
		return arr;
	}

	// Собирает по одному простому значению в вектор
	template<typename T, ENABLE_IF_FUNDAMENTAL(T)>
	std::vector<T> gather(const T& value, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		int rank, size;
		T *recvbuf = nullptr;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);
		if (rank == root) {
			recvbuf = new T[size];
		}
		auto type = get_mpi_datatype<T>();
		MPI_Gather(&value, 1, type, recvbuf, 1, type, root, comm);
		std::vector<T> result{};
		if (rank == root)
			result.assign(recvbuf, recvbuf + size);
		return result;
	}

	// Собирает со всех процессов необходимое кол-во данных
	template<typename T, ENABLE_IF_VECTOR(T)>
	T gather(const T& slice, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		typedef typename T::value_type inner;
		int rank, size, 
			recvbufSize = 0,        
			*displs     = nullptr,  // Смещения в буфере данных для i-го фрагмента
			*recvcounts = nullptr;  // Кол-во элементов, которые необходимо получить от процессов
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		// Получаем размер слайса текущего процесса
		int sliceLen = slice.size();
		// Инициализация промежуточных буферов 
		if (rank == root) {
			recvcounts = new int[size];
			displs = new int[size];
		}
		// Собираем данные о длине каждого слайса
		MPI_Gather(&sliceLen, 1, MPI_INT, recvcounts, 1, MPI_INT, root, comm);
		// Буффер для сбора данных
		inner *recvbuf = nullptr;     
		// Рассчитываем смещения для данных
		if (rank == root) {
			displs[0] = 0;
			for (auto p = 1; p < size; p++) {
				displs[p] = displs[p - 1] + recvcounts[p - 1];
			}
			// Надеюсь я правильно это посчитал
			recvbufSize = displs[size - 1] + recvcounts[size - 1];
			recvbuf = new inner[recvbufSize];
		}
		// ...
		auto type = get_mpi_datatype<inner>();
		// Собираем данные в выходной буфер
		MPI_Gatherv(&slice[0], sliceLen, type, recvbuf, recvcounts, displs, type,
			root, comm);

		// Итоговый вектор
		T result{}; 
		// Copy & CleanUp
		if (rank == root) {
			result.assign(recvbuf, recvbuf + recvbufSize);
			delete[] displs;
			delete[] recvcounts;
		}
		return result;
	}

	// ...
	template<typename T, ENABLE_IF_SARRAY(T)>
	T gather(const T& slice, int root, MPI_Comm comm = MPI_COMM_WORLD)
	{
		typedef typename T::value_type inner;
		int rank, size,
			recvbufSize = 0,
			*displs = nullptr,  // Смещения в буфере данных для i-го фрагмента
			*recvcounts = nullptr;  // Кол-во элементов, которые необходимо получить от процессов
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		// Получаем размер слайса текущего процесса
		int sliceLen = slice.size();
		// Инициализация промежуточных буферов 
		if (rank == root) {
			recvcounts = new int[size];
			displs = new int[size];
		}
		// Собираем данные о длине каждого слайса
		MPI_Gather(&sliceLen, 1, MPI_INT, recvcounts, 1, MPI_INT, root, comm);
		// Буффер для сбора данных
		inner *recvbuf = nullptr;
		// Рассчитываем смещения для данных
		if (rank == root) {
			displs[0] = 0;
			for (auto p = 1; p < size; p++) {
				displs[p] = displs[p - 1] + recvcounts[p - 1];
			}
			// Надеюсь я правильно это посчитал
			recvbufSize = displs[size - 1] + recvcounts[size - 1];
			recvbuf = new inner[recvbufSize];
		}
		// ...
		auto type = get_mpi_datatype<inner>();
		// Собираем данные в выходной буфер
		MPI_Gatherv(slice.get(), sliceLen, type, recvbuf, recvcounts, displs, type,
			root, comm);

		// Итоговый массив
		T result{};
		// Copy & CleanUp
		if (rank == root) {
			result.assign(recvbuf, recvbufSize);
			delete[] displs;
			delete[] recvcounts;
		}
		return result;
	}
}
