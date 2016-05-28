#pragma once
#include <vector>
#include "random.h"

namespace sequential {
	template <typename T, typename Container = std::vector<T>>
	size_t partition(Container& array, size_t pivot, size_t left, size_t right) {
		T value = array[pivot];
		std::swap(array[pivot], array[right]);
		auto store = left;
		for (auto i = left; i < right; ++i) {
			if (array[i] <= value) {
				std::swap(array[i], array[store]);
				store++;
			}
		}
		std::swap(array[store], array[right]);
		return store;
	}

	template <typename T, typename Container = std::vector<T>>
	void quicksort(Container& array, size_t left, size_t right) {
		if (left < right) {
			// Выбираем случайным образом оп. элемент
			int pivot = mpi::random::integer(left, right);
			// Разделяем массив на две части по опороному элементу, 
			// а также возвращаем новый опорный элемент
			size_t new_pivot = partition<T>(array, pivot, left, right);
			quicksort<T>(array, left, new_pivot - 1);
			quicksort<T>(array, new_pivot + 1, right);
		}
	}
}