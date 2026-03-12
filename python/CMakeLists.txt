project(acquire-zarr-py)

# Development.Module doesn't require a .so which is missing in manylinux containers
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)


if(NOT DEFINED pybind11_DIR)
    # pybind11 may be in a non-standard location, so we try to find it using the pybind11 module
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        RESULT_VARIABLE pybind11_NOT_FOUND
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(pybind11_NOT_FOUND)
        message(FATAL_ERROR "pybind11 not found in the current environment. Please install pybind11 via pip.")
    endif()
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(acquire_zarr acquire-zarr-py.cpp)

target_include_directories(acquire_zarr PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(acquire_zarr PRIVATE acquire-zarr)

set_target_properties(acquire_zarr PROPERTIES
    OUTPUT_NAME "__init__"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
