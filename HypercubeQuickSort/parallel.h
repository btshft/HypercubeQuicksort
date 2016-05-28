#pragma once
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include <bitset>
#include "mpiext.h"
#include "shared_array.h"

#define with(decl) \
for (bool __f = true; __f; ) \
for (decl; __f; __f = false)

namespace mpi {
	using std::vector;
	using std::shared_ptr;
	using std::pair;

	template<typename T> class sorter {

	private:
		static std::bitset<3> bin(T num){ return std::bitset<3>(num); }

	public:
		///<summary>
		/// Параллельная сортировка массива данных
		///</summary>
		static void sort(shared_array<T>& data) {
			auto slice = split(data);
			qsortpart(slice);
			data = collect(slice);
		}
	private:

		///<summary>
		/// Выбор опорной точки
		///</summary>
		static T select_pivot(shared_array<T>& data) {
			std::sort(std::begin(data), std::end(data));
			return data[data.size() / 2];
		}

		///<summary>
		/// Слияние двух массивов в один
		///</summary>
		static void merge(shared_array<T>& result, const shared_array<T>& one, const shared_array<T>& two)
		{
			// Аллокация новой памяти как общий размер двух массивов
			result.reallocate(one.size() + two.size());
			// Счетчик для перемещения по оригинальному массива
			size_t k = 0;
			// Получаем данные из первого массив
			for(size_t i = 0; i < one.size(); i++)
				result[k++] = one[i];
			// Получаем данные из второго массива
			for(size_t i = 0; i < two.size(); i++)
				result[k++] = two[i];
			// Сортировка полученного массива
			std::sort(std::begin(result), std::end(result));
		}

		///<summary>
		/// Разделение массива на две части 
		/// highpart - элементы больше опорного
		/// lowpart  - элементы меньше опорного
		///</summary>
		static void partition(const T pivot, const shared_array<T>& data,
			 shared_array<T>& lowPart, shared_array<T>& highPart)
		{
			auto high = 0,
				 low  = 0;
			// Считаем размеры для массивов low / high
			for(auto i = 0; i < data.size(); i++)
				(data[i] < pivot) ? low++ : high++;
			// Заново инициализируем массивы
			lowPart.reallocate(low);
			highPart.reallocate(high);
			// Записываем значения в массивы
			for(auto i = 0, h = 0, l = 0; i < data.size(); i++)
				(data[i] < pivot) 
					? lowPart[l++]  = data[i]
					: highPart[h++] = data[i];
		}

		///<summary>
		/// Обмен данными с соседним процессом не текущей итерации
		///</summary>
		static void exchange(shared_array<T>& data, const int iteration, const bool isHighPart)
		{
			// Определяем ранг текущего процесса и соседа
			// для отправки данных
			int rank = mpi::getRank(MPI_COMM_WORLD),
				neighbor = rank ^ (0x1 << iteration - 1),
				size = data.size(), nsize = 0;
			// Обмен массивами
			data = mpi::sendreceive(data, neighbor, neighbor, 666);
		}

		///<summary>
		/// Отправка и получение опорного элемента
		///</summary>
		static void diffusion(T& pivot, int iteration)
		{
			int rank = mpi::getRank(MPI_COMM_WORLD),
				size = mpi::getSize(MPI_COMM_WORLD);
			int root = (iteration == log2(size))
				? 0 : ((rank >> iteration) << iteration);
			int relative = rank - root;

			for(auto k = 0; k < iteration; k++) {
				if (relative < (0x1 << k)) {
					// Отправка опорного элемента
					mpi::send(pivot, rank + (0x1 << k), 666);
				}
				else if (relative < (0x1 << (k + 1))) {
					// Получение опорного элемента
					pivot = mpi::receive<T>(rank - (0x1 << k), 666);
				}
			}
		}

		///<summary>
		/// Итеративная часть алгоритма параллельной сортировки
		///</summary>
		static void qsortpart(shared_array<T>& slice)
		{
			auto rank = mpi::getRank(MPI_COMM_WORLD),
				 size = mpi::getSize(MPI_COMM_WORLD);

			// Размерность гиперкуба
			int dim  = log2(size),
				slen = slice.size();
			// Опорная точка
			T pivot = 0;
			// Массивы значений > и < чем опорный
			shared_array<T> highPart{}, lowPart{};
			//
			for(auto i = dim; i > 0; i--) {

				// Выбираем опорную точку
				if (slen != 0) {
					pivot = select_pivot(slice);
				}

				// Рассылаем её соседним процессам
				// на текущей итерации
				diffusion(pivot, i);

				// Разбиваем исходный массив на части
				// больше и меньше опорного элемента
				partition(pivot, slice, lowPart, highPart);

				// Обмен частями массива с соседними
				// элементами
				if (!(rank >> (i - 1) & 0x1)) {
					exchange(highPart, i, true);
				} else {
					exchange(lowPart, i, false);
				}
				// Слияние частей в новый массив 
				merge(slice, highPart, lowPart);
			}
		}

		///<summary>
		/// Сбор собственных частей массива в корневой процесс
		///</summary>
		static shared_array<T> collect(const shared_array<T>& slice)
		{
			return mpi::gather(slice, 0);
		}

		///<summary>
		/// Разбиение массива на N частей и отправка каждой части
		/// собственному процессу
		///</summary>
		static shared_array<T> split(shared_array<T>& data)
		{
			T* raw = data.get();
			auto slices = slice(raw, raw + data.size(), mpi::getSize(MPI_COMM_WORLD));
			auto groups = distance(slices);
			return mpi::scatter(data, groups, 0);
		}

		/// <summary>
		/// Разрезает массив на N групп
		/// </summary>
		template<typename It>
		static vector<pair<It, It>> slice(It range_from, It range_to, const ptrdiff_t num)
		{
			using diff_t = ptrdiff_t;
			// Кол-во и размер слайсов 
			const diff_t total{ std::distance(range_from, range_to) };
			const diff_t portion{ total / num };
			// Результирующий вектор
			vector<pair<It, It>> slices(num);
			// Указатель на конец слайса
			It portion_end{ range_from };
			// Использование алгоритма 'generate' для создания слайсов
			std::generate(std::begin(slices), std::end(slices), [&portion_end, portion]
			{
				// Указатель на начало текущего слайса
				It portion_start{ portion_end };
				// Обработка слайса
				std::advance(portion_end, portion);
				return std::make_pair(portion_start, portion_end);
			});
			// Указатель на конец для последней порции всегда должен указывать
			// на range_to
			slices.back().second = range_to;
			return slices;
		}

		/// <summary>
		/// Группирует массив слайсов в целочисленный вектор
		/// размера slices.size(), где каждый элемент соответствует
		/// размеру i-го слайса
		/// </summary>
		template<typename It>
		static vector<int> distance(vector<pair<It, It>>& slices)
		{
			vector<int> distances{};
			distances.reserve(slices.size());
			for (const auto& slice : slices)
				distances.push_back(std::distance(slice.first, slice.second));
			return distances;
		}

	public:
		// Класс статический, так что удаляем это всё
		sorter() = delete;
		sorter(sorter&) = delete;
		sorter(sorter&&) = delete;
		sorter& operator=(const sorter&) = delete;
	};

}
