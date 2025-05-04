cmake_minimum_required(VERSION 3.10)
project(IntegratedLLM)
set(CMAKE_CXX_STANDARD 17)

# Add llama.cpp as a subdirectory
add_subdirectory(llama.cpp)

# Create your shared library
add_library(integrated_llm SHARED
    integrated_llm.cpp
    integrated_llm.h
)

# Make sure your library depends on llama.cpp
target_link_libraries(integrated_llm PRIVATE llama)

# Include llama.cpp headers
target_include_directories(integrated_llm PRIVATE llama.cpp)

# Platform specific: Enable OpenMP if available
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(integrated_llm PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install rules (optional)
install(TARGETS integrated_llm DESTINATION lib)
install(FILES integrated_llm.h DESTINATION include)
