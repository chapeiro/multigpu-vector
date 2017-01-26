#include "union_all_cpu.cuh"

__host__ union_all_cpu::union_all_cpu(h_operator_t * parent, int children): parent(parent), children(children), children_closed(0), children_opened(0){}

__host__ void union_all_cpu::consume(buffer_pool<int32_t>::buffer_t * data){
    parent->consume(data);
}

__host__ void union_all_cpu::close(){
    m.lock();
    if (++children_closed == children) parent->close();
    m.unlock();
}

__host__ void union_all_cpu::open(){
    m.lock();
    if (++children_opened == 1) parent->open();
    m.unlock();
}