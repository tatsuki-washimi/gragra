cmake_minimum_required(VERSION 3.18)
project(gragra LANGUAGES NONE)

option(GRAGRA_BUILD_CPP "Build gragra C++ backend" OFF)

if(GRAGRA_BUILD_CPP)
  enable_language(CXX)

  # Modern C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(pybind11 CONFIG REQUIRED)
  find_package(OpenMP)

  pybind11_add_module(_cpp_backend MODULE cpp/inverse_square_backend.cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(_cpp_backend PRIVATE OpenMP::OpenMP_CXX)
  endif()

  install(TARGETS _cpp_backend DESTINATION gragra)
endif()
