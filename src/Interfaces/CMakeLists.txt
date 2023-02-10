if(ENABLE_PYTHON)
    nanobind_add_module(
        apes_interface
        NB_STATIC
        nanobind.cc
    )
    target_link_libraries(apes_interface PUBLIC project_options Channel)
    install(TARGETS apes_interface LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib)
endif()

if(ENABLE_TENSORFLOW)
    add_library(apes_tf Tensorflow.cc)
    target_link_libraries(apes_tf PRIVATE project_options PUBLIC tensorflow Channel)
    install(TARGETS apes_tf LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib)
endif()
