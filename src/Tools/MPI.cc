#include "Tools/MPI.hh"
#include "spdlog/spdlog.h"

using chili::MPIHandler;

MPIHandler MPIHandler::mpi;

void MPIHandler::Init(int argc, char* argv[]) {
#ifdef ENABLE_MPI
    MPI_Init(&argc, &argv);
    mComm = MPI_COMM_WORLD;
#endif
}

void MPIHandler::PrintRankInfo() {
#ifdef ENABLE_MPI
    const int size = Size();
    if(size > 1)
        spdlog::info("Running on {} ranks.", size);
#endif
}
