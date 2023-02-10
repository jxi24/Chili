#include "tensorflow/core/framework/shape_inference.h"
#include "Interfaces/Tensorflow.hh"
#include "Channel/MultiChannel.hh"
#include "Tools/FourVector.hh"
#include <stdexcept>
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace apes::tensorflow {

namespace functor {

template<typename Device>
struct ApesFunctor {
    void operator()(const ::tensorflow::OpKernelContext*, size_t, size_t, const double*, double*, std::unique_ptr<apes::Integrand<apes::FourVector>>&);
};

}
}

REGISTER_OP("CallApes")
    .Input("random: float64")
    .Attr("runcard: string = ''")
    .Output("xsec: float64");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

struct SingleThread {};

template<>
struct apes::tensorflow::functor::ApesFunctor<CPUDevice> {
    void operator()(OpKernelContext *ctx, size_t nBatch, size_t nRandom, const double *in, double *out, std::unique_ptr<apes::Integrand<apes::FourVector>> &integrand) {
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
                        std::vector<apes::FourVector> point;
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
struct apes::tensorflow::functor::ApesFunctor<SingleThread> {
    void operator()(OpKernelContext *ctx, size_t nBatch, size_t nRandom, const double *in, double *out, std::unique_ptr<apes::Integrand<apes::FourVector>> &integrand) {
        auto &mapping = integrand -> GetChannel(0).mapping;
        std::vector<double> rans(nRandom);
        for(size_t ibatch=0; ibatch<nBatch; ++ibatch) {
            for(size_t irand = 0; irand < nRandom; ++irand) {
                rans[irand] = in[ibatch*nRandom + irand];
            }
            std::vector<apes::FourVector> point;
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
class CallApesOp : public OpKernel {
    private:
        static std::unique_ptr<apes::Integrand<apes::FourVector>> integrand;

    public:
        explicit CallApesOp(OpKernelConstruction* context) : OpKernel(context) {
            std::string run_card;
            OP_REQUIRES_OK(context, context->GetAttr("runcard", &run_card));
            if(!integrand) {
                integrand = apes::python::ConstructIntegrand(run_card);

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
            apes::tensorflow::functor::ApesFunctor<Device>()(
                    context,
                    nBatch, nRandom,
                    input.data(),
                    output_tensor->flat<double>().data(),
                    integrand);
        }
};

template<typename Device>
std::unique_ptr<apes::Integrand<apes::FourVector>> CallApesOp<Device>::integrand = nullptr;

REGISTER_KERNEL_BUILDER(Name("CallApesThreaded").Device(DEVICE_CPU), CallApesOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("CallApes").Device(DEVICE_CPU), CallApesOp<SingleThread>);
