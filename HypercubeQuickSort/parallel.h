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
		/// ������������ ���������� ������� ������
		///</summary>
		static void sort(shared_array<T>& data) {
			auto slice = split(data);
			qsortpart(slice);
			data = collect(slice);
		}
	private:

		///<summary>
		/// ����� ������� �����
		///</summary>
		static T select_pivot(shared_array<T>& data) {
			std::sort(std::begin(data), std::end(data));
			return data[data.size() / 2];
		}

		///<summary>
		/// ������� ���� �������� � ����
		///</summary>
		static void merge(shared_array<T>& result, const shared_array<T>& one, const shared_array<T>& two)
		{
			// ��������� ����� ������ ��� ����� ������ ���� ��������
			result.reallocate(one.size() + two.size());
			// ������� ��� ����������� �� ������������� �������
			size_t k = 0;
			// �������� ������ �� ������� ������
			for(size_t i = 0; i < one.size(); i++)
				result[k++] = one[i];
			// �������� ������ �� ������� �������
			for(size_t i = 0; i < two.size(); i++)
				result[k++] = two[i];
			// ���������� ����������� �������
			std::sort(std::begin(result), std::end(result));
		}

		///<summary>
		/// ���������� ������� �� ��� ����� 
		/// highpart - �������� ������ ��������
		/// lowpart  - �������� ������ ��������
		///</summary>
		static void partition(const T pivot, const shared_array<T>& data,
			 shared_array<T>& lowPart, shared_array<T>& highPart)
		{
			auto high = 0,
				 low  = 0;
			// ������� ������� ��� �������� low / high
			for(auto i = 0; i < data.size(); i++)
				(data[i] < pivot) ? low++ : high++;
			// ������ �������������� �������
			lowPart.reallocate(low);
			highPart.reallocate(high);
			// ���������� �������� � �������
			for(auto i = 0, h = 0, l = 0; i < data.size(); i++)
				(data[i] < pivot) 
					? lowPart[l++]  = data[i]
					: highPart[h++] = data[i];
		}

		///<summary>
		/// ����� ������� � �������� ��������� �� ������� ��������
		///</summary>
		static void exchange(shared_array<T>& data, const int iteration, const bool isHighPart)
		{
			// ���������� ���� �������� �������� � ������
			// ��� �������� ������
			int rank = mpi::getRank(MPI_COMM_WORLD),
				neighbor = rank ^ (0x1 << iteration - 1),
				size = data.size(), nsize = 0;
			// ����� ���������
			data = mpi::sendreceive(data, neighbor, neighbor, 666);
		}

		///<summary>
		/// �������� � ��������� �������� ��������
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
					// �������� �������� ��������
					mpi::send(pivot, rank + (0x1 << k), 666);
				}
				else if (relative < (0x1 << (k + 1))) {
					// ��������� �������� ��������
					pivot = mpi::receive<T>(rank - (0x1 << k), 666);
				}
			}
		}

		///<summary>
		/// ����������� ����� ��������� ������������ ����������
		///</summary>
		static void qsortpart(shared_array<T>& slice)
		{
			auto rank = mpi::getRank(MPI_COMM_WORLD),
				 size = mpi::getSize(MPI_COMM_WORLD);

			// ����������� ���������
			int dim  = log2(size),
				slen = slice.size();
			// ������� �����
			T pivot = 0;
			// ������� �������� > � < ��� �������
			shared_array<T> highPart{}, lowPart{};
			//
			for(auto i = dim; i > 0; i--) {

				// �������� ������� �����
				if (slen != 0) {
					pivot = select_pivot(slice);
				}

				// ��������� � �������� ���������
				// �� ������� ��������
				diffusion(pivot, i);

				// ��������� �������� ������ �� �����
				// ������ � ������ �������� ��������
				partition(pivot, slice, lowPart, highPart);

				// ����� ������� ������� � ���������
				// ����������
				if (!(rank >> (i - 1) & 0x1)) {
					exchange(highPart, i, true);
				} else {
					exchange(lowPart, i, false);
				}
				// ������� ������ � ����� ������ 
				merge(slice, highPart, lowPart);
			}
		}

		///<summary>
		/// ���� ����������� ������ ������� � �������� �������
		///</summary>
		static shared_array<T> collect(const shared_array<T>& slice)
		{
			return mpi::gather(slice, 0);
		}

		///<summary>
		/// ��������� ������� �� N ������ � �������� ������ �����
		/// ������������ ��������
		///</summary>
		static shared_array<T> split(shared_array<T>& data)
		{
			T* raw = data.get();
			auto slices = slice(raw, raw + data.size(), mpi::getSize(MPI_COMM_WORLD));
			auto groups = distance(slices);
			return mpi::scatter(data, groups, 0);
		}

		/// <summary>
		/// ��������� ������ �� N �����
		/// </summary>
		template<typename It>
		static vector<pair<It, It>> slice(It range_from, It range_to, const ptrdiff_t num)
		{
			using diff_t = ptrdiff_t;
			// ���-�� � ������ ������� 
			const diff_t total{ std::distance(range_from, range_to) };
			const diff_t portion{ total / num };
			// �������������� ������
			vector<pair<It, It>> slices(num);
			// ��������� �� ����� ������
			It portion_end{ range_from };
			// ������������� ��������� 'generate' ��� �������� �������
			std::generate(std::begin(slices), std::end(slices), [&portion_end, portion]
			{
				// ��������� �� ������ �������� ������
				It portion_start{ portion_end };
				// ��������� ������
				std::advance(portion_end, portion);
				return std::make_pair(portion_start, portion_end);
			});
			// ��������� �� ����� ��� ��������� ������ ������ ������ ���������
			// �� range_to
			slices.back().second = range_to;
			return slices;
		}

		/// <summary>
		/// ���������� ������ ������� � ������������� ������
		/// ������� slices.size(), ��� ������ ������� �������������
		/// ������� i-�� ������
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
		// ����� �����������, ��� ��� ������� ��� ��
		sorter() = delete;
		sorter(sorter&) = delete;
		sorter(sorter&&) = delete;
		sorter& operator=(const sorter&) = delete;
	};

}
