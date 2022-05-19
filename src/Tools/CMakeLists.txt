add_library(Tools ThreeVector.cc FourVector.cc)
target_link_libraries(Tools PRIVATE project_options project_warnings
                            PUBLIC spdlog::spdlog)
