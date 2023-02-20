if(ENABLE_PYTHON)
    nanobind_add_module(
        chili_interface
        NB_STATIC
        nanobind.cc
    )
    target_link_libraries(chili_interface PUBLIC project_options Channel)
    install(TARGETS chili_interface LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib)
endif()

if(ENABLE_TENSORFLOW)
    add_library(chili_tf Tensorflow.cc)
    target_link_libraries(chili_tf PRIVATE project_options PUBLIC tensorflow Channel)
    install(TARGETS chili_tf LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib)
endif()
