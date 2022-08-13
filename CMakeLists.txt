cmake_minimum_required(VERSION 3.23.2)
project(Dual-Quaternion-Benchmarks
    VERSION 0.1
    DESCRIPTION "Benchmarks products over various dual quaternion implementations"
    HOMEPAGE_URL https://github.com/surrealwaffle/Dual-Quaternion-Benchmarks
    LANGUAGES CXX
)

add_executable(benchmark
    main.cpp
)

target_compile_features(benchmark
    PRIVATE
        cxx_std_20
)

set_target_properties(benchmark
    PROPERTIES
        CXX_EXTENSIONS OFF
)