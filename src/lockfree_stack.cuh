#ifndef OPTIMISTIC_SAFE_DEVICE_STACK_CUH_
#define OPTIMISTIC_SAFE_DEVICE_STACK_CUH_

#include "common.cuh"
#include <cstdint>
#include <limits>

using namespace std;



/**
 * Based on a combination of :
 * * https://github.com/memsql/lockfree-bench/blob/master/stack/lockfree.h
 * * (use with care) http://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
 * * http://www.boost.org/doc/libs/1_60_0/boost/lockfree/detail/freelist.hpp
 * * http://blog.memsql.com/common-pitfalls-in-writing-lock-free-algorithms/
 */
template<typename T, T invalid_value = numeric_limits<T>::max()>
class lockfree_stack{
private:
    union alignas(sizeof(uint64_t)) tagged_uint32_t{
        uint64_t pack;
        struct {
            uint32_t val;
            uint32_t tag;
        }        orig;

    public:
        __device__ __host__ tagged_uint32_t(){
        }

        __device__ __host__ tagged_uint32_t(uint64_t p): pack(p){
        }

        __device__ __host__ tagged_uint32_t(const tagged_uint32_t &t){
            pack = t.pack;
        }

        __device__ __host__ tagged_uint32_t(const volatile tagged_uint32_t &t){
            pack = t.pack;
        }

        __device__ tagged_uint32_t next() const{
            tagged_uint32_t ret;
            ret.pack = pack; //copy local prior to modifying
            ret.orig.tag++;
            return ret;
        }

        __device__ __host__ operator uint64_t() const {
            return pack;
        }

        __device__ __host__ bool operator==(const tagged_uint32_t &t) const{
            return pack == t.pack;
        }

        __device__ __host__ tagged_uint32_t operator=(uint64_t t){
            pack = t;
            return *this;
        }

        __device__ __host__ tagged_uint32_t operator=(volatile tagged_uint32_t &t){
            pack = t.pack;
            return *this;
        }
    };

    struct node_t{
        int next;
        T   data;
    };

    volatile tagged_uint32_t data_head;
    volatile tagged_uint32_t free_head;

    node_t volatile * node;

public:
    __host__ lockfree_stack(size_t size, vector<T> fill, int device){
        set_device_on_scope d(device);
        node = (node_t *) malloc((size+2)*sizeof(node_t));
        node[0].data = invalid_value;
        node[0].next = 0;
        node[1].data = invalid_value;
        node[1].next = 1;
        data_head.orig.val = 0;
        data_head.orig.tag = 0;
        free_head.orig.val = 1;
        free_head.orig.tag = 1;

        for (size_t i = 2 ; i < fill.size() + 2; ++i) {
            node[i].next       = data_head.orig.val;
            node[i].data       = fill[i-2];
            data_head.orig.tag = data_head.orig.tag + 1;
            data_head.orig.val = i;
        }
        for (size_t i = fill.size() + 2 ; i < size + 2 ; ++i) {
            node[i].next       = free_head.orig.val;
            free_head.orig.tag = free_head.orig.tag + 1;
            free_head.orig.val = i;
        }
        node_t * tmp_list;
        gpu(cudaMalloc(&tmp_list, (size+2)*sizeof(node_t)));
        gpu(cudaMemcpy(tmp_list, (node_t *) node, (size+2)*sizeof(node_t), cudaMemcpyDefault));
        free((node_t *) node);
        node = tmp_list;
    }

    __host__ ~lockfree_stack(){
        gpu(cudaFree((node_t *) node));
    }

private:
    __device__ int get_free(){ //blocking
        assert(__popc(__ballot(1)) == 1);
        tagged_uint32_t head;
        tagged_uint32_t n_head;
        tagged_uint32_t old_head;
        do {
            do head = free_head; while (head.orig.val == 1);
            assert(head.orig.val != 1);
            assert(head.orig.val != 0);
            assert(head.orig.val >  1);
            n_head.orig.val = node[head.orig.val].next;
            n_head.orig.tag = head.orig.tag + 1;
            assert(n_head.orig.val >= 1);
            old_head        = atomicCAS((uint64_t *) &free_head.pack, head.pack, n_head.pack);
        } while (head != old_head);

        assert(head.orig.val > 1);
        assert(head.orig.val < 1024+2);
        return head.orig.val;
    }

    __device__ void release_node(int index){
        assert(__popc(__ballot(1)) == 1);
        assert(index > 1);
        assert(index < 1024+2);
        tagged_uint32_t head = free_head;
        tagged_uint32_t n_head;
        while(true) {
            node[index].next = head.orig.val;
            n_head.orig.tag  = head.orig.tag + 1;
            n_head.orig.val  = index;
            assert(n_head.orig.val >= 1);
            tagged_uint32_t old_head = atomicCAS((uint64_t *) &free_head.pack, head.pack, n_head.pack);
            if (old_head == head) break;
            head = old_head;
        }
    }

public:
    __device__ void push(T v){
        assert(__popc(__ballot(1)) == 1);
        // get a free node
        int index = get_free();
        node[index].data = v;

        // add to the stack
        tagged_uint32_t head = data_head;
        tagged_uint32_t n_head;
        while(true) {
            node[index].next = head.orig.val;
            n_head.orig.tag  = head.orig.tag + 1;
            n_head.orig.val  = index;
            tagged_uint32_t old_head = atomicCAS((uint64_t *) &data_head.pack, head.pack, n_head.pack);
            if (old_head == head) break;
            head = old_head;
        }
    }

    __device__ bool try_pop(T *ret){ //blocking (as long as the stack is not empty)
        assert(__popc(__ballot(1)) == 1);
        tagged_uint32_t head = data_head;
        tagged_uint32_t n_head;

        while (head.orig.val != 0){
            n_head.orig.val = node[head.orig.val].next;
            n_head.orig.tag = head.orig.tag + 1;
            tagged_uint32_t old_head = atomicCAS((uint64_t *) &data_head.pack, head.pack, n_head.pack);
            if (head == old_head) {
                int head_index = head.orig.val;
                *ret = node[head_index].data;
                release_node(head_index);
                return true;
            }
            head = old_head;
        }
        return false;
    }

    __device__ T pop(){ //blocking
        assert(__popc(__ballot(1)) == 1);
        T ret;
        while (!try_pop(&ret));
        return ret;
    }

    __host__ __device__ static bool is_valid(T &x){
        return x != invalid_value;
    }

    __host__ __device__ static T get_invalid(){
        return invalid_value;
    }
};

#endif /* OPTIMISTIC_SAFE_DEVICE_STACK_CUH_ */