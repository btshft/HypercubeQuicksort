#pragma once
#include <vector>
#include <memory>

namespace mpi {
	using std::vector;
	using std::shared_ptr;

	template<typename T> class shared_array
	{
	private:
		struct deleter { void operator()(T* p) { delete[] p; } };
	private:
		shared_ptr<T> _data;
		size_t _size;
	public:
		// 
		shared_array() :_data(nullptr, deleter{}), _size(0) { };
		//
		shared_array(T* ndata, size_t nsize)
			: _data(ndata, deleter{}), _size(nsize)
		{ }
		//
		explicit shared_array(size_t nsize) : _size(nsize) {
			T* block = new T[nsize];
			std::memset(block, 0, nsize * sizeof(T));
			_data = std::shared_ptr<T>(block, deleter{});
		}

	public:
		// Тип значения  
		typedef T value_type;
		// Возвращает указатель на данные
		T* get() const { return _data.get(); }
		// Возвращает размер
		size_t size() const { return _size; }
		// Возвращает shared_ptr
		shared_ptr<T> getShared() const { return _data; }
	public:
		// [Works]
		T  operator[](size_t i) const
			{ return _data.get()[i]; }
		// [Works]
		T& operator[](size_t i)
			{ return _data.get()[i]; }
		// [Works] [Not-tested well]
		shared_array<T>& operator=(const shared_array<T>& nheap) {
			if (this == &nheap || this->_data == nheap.getShared())
				return *this;
			this->_size = nheap.size();
			_data = nheap.getShared();
			return *this;
		}
	public:
		// shared_array -> vector
		static vector<T> asvector(const shared_array<T>& narr) {
			return vector<T>(std::begin(narr), std::end(narr));
		}
		// vector -> shared_array
		static shared_array<T> fromvector(vector<T>& vec) {
			auto nsize = vec.size();
			T* ndata = new T[nsize];
			std::memcpy(ndata, vec.data(), vec.size() * sizeof(T));
			return shared_array<T>(ndata, nsize);
		}

	public:
		// Ресайз массива с переносом данных
		void resize(size_t nsize) {
			T* nblock = new T[nsize];
			for (size_t i = 0; i < _size; i++)
				nblock[i] = _data[i];
			_data.reset(nblock);
			_size = nsize;
		}
		// Ресайз массива без переноса данных
		void reallocate(size_t nsize) {
			T* nblock = new T[nsize];
			_data.reset(nblock);
			_size = nsize;
		}
		// Аналог конструктора
		void assign(T* data, size_t size) {
			_data.reset(data);
			_size = size;
		}
	};
};

namespace std
{
	template<typename T>
	T* begin(const mpi::shared_array<T>& arr) { return arr.get(); }
	template<typename T>
	T* end(const mpi::shared_array<T>& arr) { return arr.get() + arr.size(); }
}