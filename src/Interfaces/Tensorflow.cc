#include "Tools/MPI.hh"
#include "tensorflow/core/framework/shape_inference.h"
#include "Interfaces/Tensorflow.hh"
#include "Channel/MultiChannel.hh"
#include "Tools/FourVector.hh"
#include <stdexcept>
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace chili::tensorflow {

namespace functor {

template<typename Device>
struct ChiliFunctor {
    void operator()(const ::tensorflow::OpKernelContext*, size_t, size_t, const double*, double*, std::unique_ptr<chili::Integrand<chili::FourVector>>&);
};

}
}

REGISTER_OP("CallChili")
    .Input("random: float64")
    .Attr("runcard: string = ''")
    .Output("xsec: float64");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

struct SingleThread {};

template<>
struct chili::tensorflow::functor::ChiliFunctor<CPUDevice> {
    void operator()(OpKernelContext *ctx, size_t nBatch, size_t nRandom, const double *in, double *out, std::unique_ptr<chili::Integrand<chili::FourVector>> &integrand) {
        auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;

        thread_pool->ParallelFor(
                nBatch, nBatch*1000, // the size*1000 is cost and not sure what is the best option
                [&in, &out, &nBatch, &nRandom, &integrand](size_t start_index, size_t end_index) {
                    auto &mapping = integrand -> GetChannel(0).mapping;
                    std::vector<double> rans(nRandom);
                    for(size_t ibatch=start_index; ibatch<end_index; ++ibatch) {
                        for(size_t irand = 0; irand < nRandom; ++irand) {
                            rans[irand] = in[ibatch*nRandom + irand];
                        }
                        std::vector<chili::FourVector> point;
                        mapping -> GeneratePoint(point, rans); 

                        if(!integrand->PreProcess()(point)) {
                            out[ibatch] = 0;
                            continue;
                        }

                        double weight = mapping -> GenerateWeight(point, rans);
                        double val = weight == 0 ? 0 : integrand -> operator()(point)*weight;

                        if(!integrand->PostProcess()(point, val)) {
                            val = 0;
                        }

                        out[ibatch] = val;
                    }
                });
    }
};

template<>
struct chili::tensorflow::functor::ChiliFunctor<SingleThread> {
    void operator()(OpKernelContext *ctx, size_t nBatch, size_t nRandom, const double *in, double *out, std::unique_ptr<chili::Integrand<chili::FourVector>> &integrand) {
        auto &mapping = integrand -> GetChannel(0).mapping;
        std::vector<double> rans(nRandom);
        for(size_t ibatch=0; ibatch<nBatch; ++ibatch) {
            for(size_t irand = 0; irand < nRandom; ++irand) {
                rans[irand] = in[ibatch*nRandom + irand];
            }
            std::vector<chili::FourVector> point;
            mapping -> GeneratePoint(point, rans); 

            if(!integrand->PreProcess()(point)) {
                out[ibatch] = 0;
                continue;
            }

            double weight = mapping -> GenerateWeight(point, rans);
            double val = weight == 0 ? 0 : integrand -> operator()(point)*weight;

            if(!integrand->PostProcess()(point, val)) {
                val = 0;
            }

            out[ibatch] = val;
        }
    }
};

template<typename Device>
class CallChiliOp : public OpKernel {
    private:
        static std::unique_ptr<chili::Integrand<chili::FourVector>> integrand;

    public:
        explicit CallChiliOp(OpKernelConstruction* context) : OpKernel(context) {
            std::string run_card;
            OP_REQUIRES_OK(context, context->GetAttr("runcard", &run_card));
            if(!integrand) {
                integrand = chili::python::ConstructIntegrand(run_card);

                if(integrand -> NChannels() != 1)
                    throw std::runtime_error("TF Interface only valid for single channel");
            }
        }

        void Compute(OpKernelContext* context) override {
            // Grad the input tensor and its shape
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<double>();
            TensorShape input_shape = input_tensor.shape();
            size_t nBatch = input_shape.dim_size(0);
            size_t nRandom = input_shape.dim_size(1);

            // Create an output tensor 
            Tensor* output_tensor = NULL;
            TensorShape output_shape = input_shape;
            output_shape.RemoveDim(1);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            // Calculate the cross-sections
            chili::tensorflow::functor::ChiliFunctor<Device>()(
                    context,
                    nBatch, nRandom,
                    input.data(),
                    output_tensor->flat<double>().data(),
                    integrand);
        }
};

template<typename Device>
std::unique_ptr<chili::Integrand<chili::FourVector>> CallChiliOp<Device>::integrand = nullptr;

REGISTER_KERNEL_BUILDER(Name("CallChiliThreaded").Device(DEVICE_CPU), CallChiliOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("CallChili").Device(DEVICE_CPU), CallChiliOp<SingleThread>);
