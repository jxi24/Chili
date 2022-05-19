add_subdirectory(Tools)
add_subdirectory(Integrator)
add_subdirectory(Channel)

add_executable(example example.cc)
target_link_libraries(example PRIVATE project_options project_warnings
                              PUBLIC Channel)
