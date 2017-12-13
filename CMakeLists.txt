SET(CUDA_SEPARABLE_COMPILATION ON)
SET(POSITION_INDEPENDENT_CODE ON)

# find_package(CUDA REQUIRED)
# set(CUDA_HOST_COMPILER "g++") #FIXME: workaround as nvcc does not support the current clang version

file(GLOB multigpu_src
    # src/operators/*.cu
    # src/buffer_pool.cu
    # src/output_composer.cu
    src/buffer_manager.cu
    src/common.cu
    )


# list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/output_composer.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/select.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/select2.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/materializer.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/operator.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/aggregation.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/select3.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/select_v.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/union_all_cpu.cu)

add_library(multigpu SHARED ${multigpu_src} OPTIONS ${CUDA_CXXFLAGS})
# set_target_properties(multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(multigpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(multigpu PROPERTIES COMPILE_FLAGS "${CXXFLAGS}")
target_link_libraries(multigpu jsmn)

install(TARGETS multigpu
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
