cmake_minimum_required(VERSION 3.16)

project(SovereignQPSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sovereign_qp
    src/qp_model.cpp
    src/qp_solver.cpp
)

target_include_directories(sovereign_qp PUBLIC include)

add_executable(qp_demo
    examples/basic_qp.cpp
)

target_link_libraries(qp_demo PRIVATE sovereign_qp)

add_executable(qp_tests
    tests/qp_tests.cpp
)

target_link_libraries(qp_tests PRIVATE sovereign_qp)

enable_testing()

add_test(NAME qp_tests COMMAND qp_tests)