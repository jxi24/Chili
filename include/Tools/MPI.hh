#pragma once

#include <type_traits>
#include "Integrator/Statistics.hh"

#ifdef ENABLE_MPI
#include "mpi.h"
#endif

namespace apes {

#ifdef ENABLE_MPI

template<typename T>
struct type_builder {
    
};
#endif

namespace detail {

template<typename T, typename = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<T, std::void_t<decltype(std::declval<T>().data()),
                                   decltype(std::declval<T>().size())>> : std::true_type {};

// #ifdef ENABLE_MPI
// template<typename T, typename Enable = void>
// struct datatype_traits_impl {
//     static MPI_Datatype type() {
//         static type_builder<T> builder;
//         return builder.type_;
//     }
// };
// #endif
}

// #ifdef ENABLE_MPI
// template<typename T>
// struct datatype_traits {
//     static MPI_Datatype type() {
//         return datatype_traits_impl<T>::type();
//     }
// };
// 
/*
#define MPI_BUILTIN_TYPE(type, mpi_type)	  \
template<> struct datatype_traits<type> {	  \
  static MPI_Datatype type() { return mpi_type; } \
}

MPI_BUILTIN_TYPE(double, MPI_DOUBLE);

#undef MPI_BUILTIN_TYPE
#endif
*/

#ifdef ENABLE_MPI
struct Add {
    static void call(double *in, double *inout, int *len, MPI_Datatype*) {
        for(int i = 0; i < *len; ++i) {
            inout[i] += in[i];
        }
    }
};

struct StatsAdd {
    static void call(apes::StatsData *in, apes::StatsData *inout, int *len, MPI_Datatype*) {
        for(int i = 0; i < *len; ++i) {
            inout[i] += in[i];
        }
    }
};
#endif

class MPIHandler {
    private:
        MPIHandler() = default;
        static MPIHandler mpi;

    public:
        void Init(int argc, char* argv[]);
        static MPIHandler& Instance() {
            return mpi;
        }

        void PrintRankInfo();

#ifdef ENABLE_MPI
        void CleanUp() { MPI_Finalize(); }
        void SetWorld(const MPI_Comm &comm) { mComm = comm; }

        void Barrier() {
            MPI_Barrier(mComm);
        }

        int Rank() {
            int rank;
            MPI_Comm_rank(mComm, &rank);
            return rank;
        }

        int Size() {
            int size;
            MPI_Comm_size(mComm, &size);
            return size;
        }

        void Broadcast(void *buffer, int count, MPI_Datatype type, int root=0) {
            MPI_Bcast(buffer, count, type, root, mComm);
        }

        void Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root=0) {
            MPI_Scatter(sendbuf, sendcount, sendtype,
                        recvbuf, recvcount, recvtype,
                        root, mComm);
        }

        void Scatterv(void* sendbuf, int* sendcount, int* displace, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root=0) {
            MPI_Scatterv(sendbuf, sendcount, displace, sendtype,
                         recvbuf, recvcount, recvtype,
                         root, mComm);
        }

        void Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                    void* recvbuf, int recvcount, MPI_Datatype recvtype,
                    int root=0) {
            MPI_Gather(sendbuf, sendcount, sendtype,
                       recvbuf, recvcount, recvtype,
                       root, mComm);
        }

        void AllReduce(void* sendbuf, int sendcount, MPI_Datatype sendtype, MPI_Op op) {
	  MPI_Allreduce(MPI_IN_PLACE, sendbuf, sendcount, sendtype, op, mComm);
        }

        template<typename T, typename Op>
        void AllReduce(T &buffer) {
            auto op = Operator<Op>();
            if constexpr(detail::is_container<T>()) {
                auto type = Type<typename T::value_type>();
                MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), type, op, mComm);
            } else {
                auto type = Type<T>();
                MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, type, op, mComm);
            }
        }

        template<typename T>
        void RegisterType(MPI_Datatype type) {
            MPI_Type_commit(&type);
            types[typeid(T).name()] = type;
        }

        template<typename Op>
        void RegisterOp() {
            MPI_Op mpi_op;
            MPI_Op_create(reinterpret_cast<MPI_User_function*>(Op::call), true, &mpi_op);
            operators[typeid(Op).name()] = mpi_op;
        }

        template<typename T>
        MPI_Datatype Type() const {
            const auto name = typeid(T).name();
            if(types.find(name) != types.end())
                return types.at(name);
            throw std::runtime_error("Invalid MPI Type. Please register it");
        }

        template<typename Op>
        MPI_Op Operator() const {
            const auto name = typeid(Op).name();
            if(operators.find(name) != operators.end())
                return operators.at(name);
            throw std::runtime_error("Invalid MPI Op. Please register it");
        }

        void GatherAll(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype) {
            MPI_Allgather(sendbuf, sendcount, sendtype,
                          recvbuf, recvcount, recvtype, mComm);
        }

        void GatherAllV(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int *recvcount, int* displace,
                             MPI_Datatype recvtype) {
            MPI_Allgatherv(sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, displace, recvtype, mComm);
        }

        void Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag) {
            MPI_Recv(buf, count, datatype, source, tag, mComm, MPI_STATUS_IGNORE);
        }

        int Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag) {
            return MPI_Send(buf, count, datatype, dest, tag, mComm);
        }

    private:
        MPI_Comm mComm = MPI_COMM_WORLD;
        std::map<std::string, MPI_Datatype> types;
        std::map<std::string, MPI_Op> operators;
#else
        // Dummy functions if MPI isn't installed
        class MPI_Datatype {};
        class MPI_Op {};
        void CleanUp() { }
        void Barrier() {}
        int Rank() { return 0; }
        int Size() { return 1; }
        void Broadcast(void*, int, MPI_Datatype, int=0) {}
        void Scatter(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int=0) {}
        void Scatterv(void*, int*, int*, MPI_Datatype, void*, int, MPI_Datatype, int=0) { }
        void Gather(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int=0) { }
        void AllReduce(void*, int, MPI_Datatype, MPI_Op) { }
        void GatherAll(const void*, int, MPI_Datatype, void*, int, MPI_Datatype) { }
        void GatherAllV(const void*, int, MPI_Datatype, void*, int*, int*, MPI_Datatype) { }
        void Recv(void*, int, MPI_Datatype, int, int) { }
        int Send(const void*, int, MPI_Datatype, int, int) { return 1; }
#endif
};

}
