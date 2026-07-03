# This script reads the test categories YAML file and applies labels to CTest

# Find Python3 for running the parser script
find_package(Python3 COMPONENTS Interpreter)

# Validate target_name, yaml_file, and working_dir for apply_test_category_labels.
function(_ctest_validate_gtest_category_label_args target_name yaml_file working_dir out_ok)
    set(_ok TRUE)
    if("${target_name}" STREQUAL "")
        message(WARNING "target_name is empty, cannot generate test categories")
        set(_ok FALSE)
    endif()
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(_ok FALSE)
    endif()
    if(NOT IS_DIRECTORY "${working_dir}")
        message(WARNING "Working directory does not exist: ${working_dir}")
        set(_ok FALSE)
    endif()
    set(${out_ok} ${_ok} PARENT_SCOPE)
endfunction()

# Run parse_test_categories.py and include the generated test_categories.cmake.
function(_ctest_run_gtest_category_parser parse_script out_ok)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${parse_script} ${ARGN}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    set(${out_ok} TRUE PARENT_SCOPE)
    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        set(${out_ok} FALSE PARENT_SCOPE)
        return()
    endif()

    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(WARNING "Generated test categories file not found: ${CATEGORY_CMAKE}")
        set(${out_ok} FALSE PARENT_SCOPE)
        return()
    endif()

    include("${CATEGORY_CMAKE}")
endfunction()

# Function to apply category labels to discovered GTest tests
# Optional 4th parameter: install_test_file - path to write install-time test definitions
# Optional 5th parameter: resource_group - CTest RESOURCE_GROUPS token to apply to
#   every generated category/GPU-exclusion suite (e.g. "gfx942" or "gpus"). When
#   provided, the suite names also gain a "_<resource>" segment so that the same
#   target can be wired in multiple times against different resource groups
#   without colliding on test names.
# Optional 6th parameter: extra parser flag(s) for parse_test_categories.py (for
#   example "--use-rtest-driver" to run the project's <stem>_rtest.py driver with
#   -t ctest_<category> instead of invoking the gtest binary directly).
function(apply_test_category_labels target_name yaml_file working_dir)
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    _ctest_validate_gtest_category_label_args(
        ${target_name} "${yaml_file}" "${working_dir}" _args_ok
    )
    if(NOT _args_ok)
        return()
    endif()

    set(PARSE_SCRIPT "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_test_categories.py")
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(WARNING "Test category parser script not found: ${PARSE_SCRIPT}")
        return()
    endif()

    # Use ARGN rather than ARGV<N>; unset ARGV<N> variables can fall through to
    # parent scopes when this function is called from another function.
    set(install_test_file "")
    set(resource_group "")
    set(parse_extra_flags "")
    list(POP_FRONT ARGN install_test_file resource_group parse_extra_flags)

    set(extra_args "")
    if(resource_group)
        list(APPEND extra_args "--resource-group" "${resource_group}")
    endif()
    if(NOT "${parse_extra_flags}" STREQUAL "")
        list(APPEND extra_args "${parse_extra_flags}")
    endif()
    if(Python3_EXECUTABLE)
        list(APPEND extra_args "--cmake-python3" "${Python3_EXECUTABLE}")
    endif()
    if(install_test_file)
        set(python_args ${extra_args} ${yaml_file} ${target_name} ${working_dir} ${install_test_file})
    else()
        set(python_args ${extra_args} ${yaml_file} ${target_name} ${working_dir})
    endif()

    _ctest_run_gtest_category_parser("${PARSE_SCRIPT}" _run_ok ${python_args})
    if(NOT _run_ok)
        return()
    endif()
endfunction()

# Function to apply category labels to discovered Catch2 tests using tag-based filtering
# Optional 4th parameter: install_test_file - path to write install-time test definitions
function(apply_catch2_test_category_labels target_name yaml_file working_dir)
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse Catch2 test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if("${target_name}" STREQUAL "")
        message(WARNING "target_name is empty, cannot generate Catch2 test categories")
        set(_validation_failed TRUE)
    endif()
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Catch2 test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(NOT IS_DIRECTORY "${working_dir}")
        message(WARNING "Working directory does not exist: ${working_dir}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    set(PARSE_SCRIPT "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_catch2_categories.py")
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(WARNING "Catch2 test category parser script not found: ${PARSE_SCRIPT}")
        return()
    endif()

    # Use ARGN rather than ARGV<N>; unset ARGV<N> variables can fall through to
    # parent scopes when this function is called from another function.
    set(install_test_file "")
    list(POP_FRONT ARGN install_test_file)

    if(install_test_file)
        set(python_args ${yaml_file} ${target_name} ${working_dir} ${install_test_file})
    else()
        set(python_args ${yaml_file} ${target_name} ${working_dir})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse Catch2 test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/catch2_test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated Catch2 test category configuration: ${CATEGORY_CMAKE}")

    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(WARNING "Generated Catch2 test categories file not found: ${CATEGORY_CMAKE}")
        return()
    endif()

    include("${CATEGORY_CMAKE}")
endfunction()

# Function to apply category labels to CTest tests already registered in the
# current directory (no GTest discovery required).
#
# Optional 2nd parameter: install_test_file - path to a hand-rolled
#   CTestTestfile.cmake (or fragment) that the caller has written
#   add_test() lines into. When provided, the parser additionally appends
#   the generated label code to that file AND scans it for the test names
#   to emit explicit per-test set_tests_properties() calls -- ctest's own
#   script interpreter does not implement get_property(DIRECTORY ...
#   PROPERTY TESTS), and set_property(TEST ...) labels are not visible to
#   ctest -L, so the default runtime enumeration loop is a silent no-op at
#   ctest time. When omitted (autogenerated CTestTestfile case), the parser
#   uses the directory-property loop, which CMake materialises into
#   explicit set_tests_properties() lines at configure time.
function(apply_ctest_category_labels yaml_file)
    # Execute the Python script to generate CMake code
    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found, cannot parse test categories YAML")
        return()
    endif()

    # Validate inputs
    set(_validation_failed FALSE)
    if(NOT EXISTS "${yaml_file}")
        message(WARNING "Test categories YAML file not found: ${yaml_file}")
        set(_validation_failed TRUE)
    endif()
    if(_validation_failed)
        return()
    endif()

    # Verify the parser script exists
    set(PARSE_SCRIPT
        "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_ctest_categories.py"
    )
    if(NOT EXISTS "${PARSE_SCRIPT}")
        message(
            FATAL_ERROR
            "Test category parser script not found: ${PARSE_SCRIPT}"
        )
    endif()

    # Use ARGN rather than ARGV<N>; unset ARGV<N> variables can fall through to
    # parent scopes when this function is called from another function.
    set(install_test_file "")
    list(POP_FRONT ARGN install_test_file)

    if(install_test_file)
        set(python_args ${yaml_file} ${install_test_file})
    else()
        set(python_args ${yaml_file})
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PARSE_SCRIPT} ${python_args}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Verify the generated CMake file exists before including it
    if(NOT EXISTS "${CATEGORY_CMAKE}")
        message(
            WARNING
            "Generated test categories file not found: ${CATEGORY_CMAKE}"
        )
        return()
    endif()

    # Include and execute the generated CMake code. When an
    # install_test_file was provided, the generated code is shaped for
    # ctest's interpreter (explicit test names, no get_property loop)
    # and is meant to live inside the install file; including it here
    # would just emit `if(TEST x)` guards for tests that may not exist
    # in this directory scope -- harmless, but noisy. Build-tree callers
    # (no install file) are the consumers of the include() path.
    if(NOT install_test_file)
        include("${CATEGORY_CMAKE}")
    endif()
endfunction()
