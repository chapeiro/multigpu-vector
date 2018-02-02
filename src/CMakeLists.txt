SET(CUDA_SEPARABLE_COMPILATION ON)
SET(POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER "g++") #FIXME: workaround as nvcc does not support the current clang version

file(GLOB multigpu_src
    # operators/*.cu
    # buffer_pool.cu
    # output_composer.cu
    buffer_manager.cu
    # common.cpp
    )

# set_property(SOURCE common.cu property LANGUAGE CXX)
# set_property(SOURCE buffer_manager.cu property LANGUAGE CXX)
# set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${CMAKE_CXX_SOURCE_FILE_EXTENSIONS};cu)

# list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/output_composer.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/select.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/select2.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/materializer.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/operator.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/aggregation.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/select3.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/select_v.cu)
list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/operators/union_all_cpu.cu)

get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#it seems that cuda_add_library does take into consideration the include_directories directives
#hopefully the next like will become obsolete
string(REGEX REPLACE "([^;]+)" "-I\\1" inc_dirs2 "${inc_dirs}")

cuda_add_library(multigpu STATIC ${multigpu_src} OPTIONS ${CUDA_CUFLAGS} ${inc_dirs2})
set_target_properties(multigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(multigpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(multigpu PROPERTIES COMPILE_FLAGS "${CUDA_CUFLAGS}")
target_link_libraries(multigpu)

install(TARGETS multigpu
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
