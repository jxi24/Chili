add_library(Integrator AdaptiveMap.cc Vegas.cc)
target_link_libraries(Integrator PRIVATE project_options project_warnings
                                 PUBLIC fmt::fmt yaml-cpp)
