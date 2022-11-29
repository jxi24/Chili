add_library(cudainterface Interface.cu)
set_target_properties(cudainterface PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(cudainterface PRIVATE project_options PUBLIC fmt::fmt)
