set(tgt acquire-zarr)

if (WIN32)
    set(PLATFORM_FILE_SINK_CPP win32/file.sink.impl.cpp)
else ()
    set(PLATFORM_FILE_SINK_CPP posix/file.sink.impl.cpp)
endif ()

add_library(${tgt}
        macros.hh
        acquire.zarr.cpp
        zarr.dimension.hh
        zarr.dimension.cpp
        frame.queue.hh
        frame.queue.cpp
        downsampler.hh
        downsampler.cpp
        zarr.stream.hh
        zarr.stream.cpp
        zarr.common.hh
        zarr.common.cpp
        blosc.compression.params.hh
        blosc.compression.params.cpp
        thread.pool.hh
        thread.pool.cpp
        s3.connection.hh
        s3.connection.cpp
        sink.hh
        sink.cpp
        file.sink.hh
        file.sink.cpp
        ${PLATFORM_FILE_SINK_CPP}
        s3.sink.hh
        s3.sink.cpp
        array.writer.hh
        array.writer.cpp
        zarrv2.array.writer.hh
        zarrv2.array.writer.cpp
        zarrv3.array.writer.hh
        zarrv3.array.writer.cpp
        $<TARGET_OBJECTS:acquire-logger-obj>
)

target_include_directories(${tgt}
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/logger>
)

target_link_libraries(${tgt} PRIVATE
        blosc_static
        miniocpp::miniocpp
        Crc32c::crc32c
        OpenMP::OpenMP_CXX
)

target_compile_definitions(${tgt} PRIVATE
        "ACQUIRE_ZARR_API_VERSION=\"0.4.0\""
)

set_target_properties(${tgt} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        POSITION_INDEPENDENT_CODE ON
)

install(TARGETS ${tgt}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

# Install public header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION include
)