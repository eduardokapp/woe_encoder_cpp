cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
    DESCRIPTION "Fast Weight of Evidence Encoder"
)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Set C++ standard and policies
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Core library sources
set(CPP_SOURCES
    fast_woe_encoder.cc
    main.cc
)

# Main encoder library (static library - may not be needed for pybind11 module)
add_library(fast_woe_encoder STATIC ${CPP_SOURCES})
target_include_directories(fast_woe_encoder
    PUBLIC
        ${Python_INCLUDE_DIRS} # Add Python's include directories here (potentially redundant)
        ${pybind11_INCLUDE_DIRS} # If using pybind11, include its headers too (potentially redundant)
        include # Your project's own include directory (if you have one called 'include') (potentially redundant)
)

pybind11_add_module(_core MODULE main.cc WITH_SOABI)
target_link_libraries(_core
    PRIVATE
        pybind11::headers
        fast_woe_encoder  # This line is the fix
)

# Add include directories for Python and pybind11 to the _core module target (CRUCIAL FIX)
target_include_directories(_core
    PUBLIC
        ${Python_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION fast_woe_encoder)

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()
endif()