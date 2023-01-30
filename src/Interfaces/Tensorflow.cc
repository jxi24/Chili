#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "Interfaces/Tensorflow.hh"
#include "Channel/MultiChannel.hh"
#include "Tools/FourVector.hh"
#include <stdexcept>

REGISTER_OP("CallApes")
    .Input("random: float64")
    .Attr("runcard: string")
    .Output("xsec: float64");

using namespace tensorflow;

class CallApesOp : public OpKernel {
    private:
        apes::Integrand<apes::FourVector> integrand;

    public:
        explicit CallApesOp(OpKernelConstruction* context) : OpKernel(context) {
            std::string run_card;
            OP_REQUIRES_OK(context, context->GetAttr("runcard", &run_card));
            integrand = apes::tensorflow::ConstructIntegrand(run_card);

            if(integrand.NChannels() != 1)
                throw std::runtime_error("TF Interface only valid for single channel");
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
            auto output_flat = output_tensor->flat<double>();

            // Get the channel mapping
            auto &mapping = integrand.GetChannel(0).mapping;

            // Calculate the cross-sections
            std::vector<double> rans(nRandom);
            for(size_t ibatch = 0; ibatch < nBatch; ++ibatch) {
                for(size_t irand = 0; irand < nRandom; ++irand) {
                    rans[irand] = input(ibatch*nRandom + irand);
                }
                std::vector<apes::FourVector> point;
                mapping -> GeneratePoint(point, rans); 

                if(!integrand.PreProcess()(point)) {
                    output_flat(ibatch) = 0;
                    continue;
                }

                double weight = mapping -> GenerateWeight(point, rans);
                double val = weight == 0 ? 0 : integrand(point)*weight;

                if(!integrand.PostProcess()(point, val)) {
                    val = 0;
                }

                output_flat(ibatch) = val;
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("CallApes").Device(DEVICE_CPU), CallApesOp);
