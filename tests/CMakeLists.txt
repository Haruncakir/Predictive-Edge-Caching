# tests/CMakeLists.txt ──────────────────────────────────────────────────────
# Each test is a standalone executable registered with CTest.

set(TEST_SOURCES
    test_bounded_queue.cpp
    test_history_buffer.cpp
    test_csv_loader.cpp
    test_synthetic_engine.cpp
    test_json_parser.cpp
)

# All tests need the project's include path + the test harness.
foreach(test_src ${TEST_SOURCES})
    # Strip extension to get the target name.
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Link against the same source files the main binary uses (except main.cpp).
    target_sources(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/DatasetLoader.cpp
        ${CMAKE_SOURCE_DIR}/src/TrafficEngine.cpp
        ${CMAKE_SOURCE_DIR}/src/RequestSimulator.cpp
        ${CMAKE_SOURCE_DIR}/src/SimConfig.cpp
    )

    target_link_libraries(${test_name} PRIVATE Threads::Threads)

    if(MSVC)
        target_compile_options(${test_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${test_name} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
    endif()

    # Register with CTest.  Tests that read data files need to run from
    # the project root so relative paths (data/sample_trace.csv) work.
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endforeach()
