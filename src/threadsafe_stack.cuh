#ifndef THREADSAFE_STACK_CUH_
#define THREADSAFE_STACK_CUH_

#include <cstdint>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <vector>

using namespace std;

template<typename T, T invalid_value = numeric_limits<T>::max()>
class threadsafe_stack{
private:
    vector<T> data;
    mutex     m;
    condition_variable cv;

public:
    __host__ threadsafe_stack(size_t size, vector<T> fill){
        data.reserve(size);
        for (const auto &t: fill) data.push_back(t);
    }

    __host__ ~threadsafe_stack(){}

public:
    __host__ void push(T v){
        unique_lock<std::mutex> lock(m);
        data.push_back(v);
        cv.notify_all();
    }

    __host__ bool try_pop(T *ret){ //blocking (as long as the stack is not empty)
        unique_lock<std::mutex> lock(m);
        if (data.empty()) return false;
        *ret = data.back();
        data.pop_back();
        return true;
    }

    __host__ T pop(){ //blocking
        unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this]{return !data.empty();});
        T ret = data.back();
        data.pop_back();
        return ret;
    }

    __host__ __device__ static bool is_valid(T &x){
        return x != invalid_value;
    }

    __host__ __device__ static T get_invalid(){
        return invalid_value;
    }
};

#endif /* THREADSAFE_STACK_CUH_ */