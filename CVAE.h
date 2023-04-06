//
// Created by 17110 on 2023/4/5.
//

#ifndef OPENCV_MNIST_CVAE_H
#define OPENCV_MNIST_CVAE_H

#include <torch/torch.h>


class CVAE : public torch::nn::Module {
private:
    torch::nn::Linear encoder_layer_1, encoder_layer_2, encoder_layer_3;
    torch::nn::Linear encoder_mean, encoder_log_std;
    torch::nn::Linear decoder_layer_1, decoder_layer_2, decoder_layer_3;
public:
    CVAE(int input_size, int condition_size, int latent_size);

    std::map<std::string, torch::Tensor> forward(const torch::Tensor &x, const torch::Tensor &condition);

    static torch::Tensor loss_func(const torch::Tensor &x, const torch::Tensor &recon, const torch::Tensor &mean,
                            const torch::Tensor &log_std);

    torch::Tensor generate(const torch::Tensor &condition,int latent_size);
};


#endif //OPENCV_MNIST_CVAE_H
