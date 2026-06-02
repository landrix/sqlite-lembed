cmake_minimum_required(VERSION 3.14)
project(SqliteEmbed C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(LLAMA_METAL OFF)
set(LLAMA_STATIC ON)
set(LLAMA_OPENMP OFF)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")
set(SQLITE_AMALGAMATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite")
set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION)
string(STRIP "${VERSION}" VERSION)
string(TIMESTAMP DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
execute_process(
    COMMAND git log -n 1 --pretty=format:%H -- VERSION
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE SOURCE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/sqlite-lembed.h.tmpl"
    "${GENERATED_INCLUDE_DIR}/sqlite-lembed.h"
)

include(ExternalProject)

set(SQLITE_VERSION 3450300)
set(SQLITE_YEAR 2024)
set(SQLITE_URL https://www.sqlite.org/${SQLITE_YEAR}/sqlite-amalgamation-${SQLITE_VERSION}.zip)
ExternalProject_Add(sqlite_amalgamation
    URL             ${SQLITE_URL}
    DOWNLOAD_DIR    ${CMAKE_BINARY_DIR}/downloads
    SOURCE_DIR      ${SQLITE_AMALGAMATION_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
)


add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

include_directories(${SQLITE_AMALGAMATION_DIR})

add_library(sqlite_lembed SHARED sqlite-lembed.c)
add_dependencies(sqlite_lembed sqlite_amalgamation)
target_link_libraries(sqlite_lembed ggml llama)
target_include_directories(sqlite_lembed PRIVATE ${LLAMA_CPP_DIR} ${GENERATED_INCLUDE_DIR})
set_target_properties(sqlite_lembed PROPERTIES PREFIX "")
set_target_properties(sqlite_lembed PROPERTIES OUTPUT_NAME "lembed0")

add_library(sqlite_lembed_static STATIC sqlite-lembed.c)
add_dependencies(sqlite_lembed_static sqlite_amalgamation)
target_link_libraries(sqlite_lembed_static ggml_static llama)
target_include_directories(sqlite_lembed_static PRIVATE ${LLAMA_CPP_DIR} ${GENERATED_INCLUDE_DIR})
target_compile_definitions(sqlite_lembed_static PRIVATE SQLITE_CORE)
set_target_properties(sqlite_lembed_static PROPERTIES OUTPUT_NAME "sqlite_lembed0")
