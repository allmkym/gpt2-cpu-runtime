cmake_minimum_required(VERSION 3.20)

project(gpt2_cpu_runtime VERSION 0.1.0 LANGUAGES CXX)

option(GPT2_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(GPT2_ENABLE_TSAN "Enable ThreadSanitizer (separate from ASan/UBSan)" OFF)
set(GPT2_TEST_CHECKPOINT "" CACHE FILEPATH
    "Optional real checkpoint used by the integration test")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(gpt2_runtime
    src/kernels.cpp
    src/linear_executor.cpp
    src/model.cpp
    src/session.cpp
)
target_include_directories(gpt2_runtime PUBLIC include)
target_compile_features(gpt2_runtime PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(gpt2_runtime PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpt2_runtime PRIVATE -ffp-contract=off)
endif()

if(GPT2_ENABLE_SANITIZERS AND GPT2_ENABLE_TSAN)
    message(FATAL_ERROR "ASan/UBSan and TSan cannot be enabled together")
endif()

if(GPT2_ENABLE_SANITIZERS OR GPT2_ENABLE_TSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers require GCC or Clang")
    endif()
endif()

if(GPT2_ENABLE_SANITIZERS)
    target_compile_options(gpt2_runtime PUBLIC
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(gpt2_runtime PUBLIC -fsanitize=address,undefined)
endif()

if(GPT2_ENABLE_TSAN)
    target_compile_options(gpt2_runtime PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(gpt2_runtime PUBLIC -fsanitize=thread)
endif()

add_executable(gpt2_inspect app/gpt2_inspect.cpp)
target_link_libraries(gpt2_inspect PRIVATE gpt2_runtime)

add_executable(gpt2_generate app/gpt2_generate.cpp)
target_link_libraries(gpt2_generate PRIVATE gpt2_runtime)

add_executable(gpt2_benchmark app/gpt2_benchmark.cpp)
target_link_libraries(gpt2_benchmark PRIVATE gpt2_runtime)

add_executable(gpt2_linear_benchmark app/gpt2_linear_benchmark.cpp)
target_link_libraries(gpt2_linear_benchmark PRIVATE gpt2_runtime)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

    add_executable(gpt2_unit_tests tests/gpt2_tests.cpp)
    target_link_libraries(gpt2_unit_tests PRIVATE gpt2_runtime)
    add_test(NAME gpt2_unit_tests COMMAND gpt2_unit_tests)

    add_executable(gpt2_session_tests tests/session_tests.cpp)
    target_link_libraries(gpt2_session_tests PRIVATE gpt2_runtime)
    add_test(NAME gpt2_session_tests COMMAND gpt2_session_tests)

    add_executable(gpt2_linear_executor_tests tests/linear_executor_tests.cpp)
    target_link_libraries(gpt2_linear_executor_tests PRIVATE gpt2_runtime)
    add_test(NAME gpt2_linear_executor_tests COMMAND gpt2_linear_executor_tests)

    add_test(NAME gpt2_cli_tests
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.py
                     $<TARGET_FILE:gpt2_generate>)

    add_test(NAME gpt2_compare_logits_tests
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_logits_test.py
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/compare_logits.py)

    if(EXISTS "${GPT2_TEST_CHECKPOINT}")
        add_executable(gpt2_real_checkpoint_test tests/real_checkpoint_test.cpp)
        target_link_libraries(gpt2_real_checkpoint_test PRIVATE gpt2_runtime)
        add_test(NAME gpt2_real_checkpoint
                 COMMAND gpt2_real_checkpoint_test "${GPT2_TEST_CHECKPOINT}")
        set_tests_properties(gpt2_real_checkpoint PROPERTIES TIMEOUT 300)
    elseif(GPT2_TEST_CHECKPOINT)
        message(STATUS "Real checkpoint integration test disabled: ${GPT2_TEST_CHECKPOINT} not found")
    else()
        message(STATUS "Real checkpoint integration test disabled; set GPT2_TEST_CHECKPOINT to enable it")
    endif()
endif()
