set(TEST_SRC_FILES)
foreach(ext IN LISTS SRC_FILE_EXTENSIONS)
    file(GLOB_RECURSE tmp_files "./${ext}")
    list(APPEND TEST_SRC_FILES ${tmp_files})
endforeach()
message(STATUS "TEST_SRC_FILES: ${TEST_SRC_FILES}")

set(project_test "${PROJECT_NAME}_tests")

add_executable(${project_test}
  ${TEST_SRC_FILES}
)

target_include_directories(${project_test}
    PRIVATE /usr/local/cuda/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/hal
    PRIVATE ${CMAKE_SOURCE_DIR}/include/internal/kernel
)

target_link_libraries(${project_test}
    PRIVATE
    ${project_lib_name}
    pthread
    gtest
)

add_test(NAME ${project_test} COMMAND ${project_test})