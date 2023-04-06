//
// Created by 17110 on 2023/4/5.
//

#include "CVAE.h"

#include <utility>

CVAE::CVAE(int input_size, int condition_size, int latent_size) : encoder_layer_1(nullptr), encoder_layer_2(nullptr),
                                                                  encoder_layer_3(nullptr), encoder_mean(nullptr),
                                                                  encoder_log_std(nullptr), decoder_layer_1(nullptr),
                                                                  decoder_layer_2(nullptr), decoder_layer_3(nullptr) {
    int hidden_size_list[] = {input_size / 2, input_size / 4, input_size / 8};
    encoder_layer_1 = torch::nn::Linear(input_size, hidden_size_list[0]);
    encoder_layer_2 = torch::nn::Linear(hidden_size_list[0], hidden_size_list[1]);
    encoder_layer_3 = torch::nn::Linear(hidden_size_list[1], hidden_size_list[2]);

    encoder_mean = torch::nn::Linear(hidden_size_list[2], latent_size);
    encoder_log_std = torch::nn::Linear(hidden_size_list[2], latent_size);

    auto dec_input_size = latent_size + condition_size;
    decoder_layer_1 = torch::nn::Linear(dec_input_size, hidden_size_list[1]);
    decoder_layer_2 = torch::nn::Linear(hidden_size_list[1], hidden_size_list[0]);
    decoder_layer_3 = torch::nn::Linear(hidden_size_list[0], input_size);
//    std::cout << "latent " << latent_size + condition_size << std::endl;

    register_module("encoder_layer_1", encoder_layer_1);
    register_module("encoder_layer_2", encoder_layer_2);
    register_module("encoder_layer_3", encoder_layer_3);
    register_module("encoder_mean", encoder_mean);
    register_module("encoder_log_std", encoder_log_std);
    register_module("decoder_layer_1", decoder_layer_1);
    register_module("decoder_layer_2", decoder_layer_2);
    register_module("decoder_layer_3", decoder_layer_3);
}

std::map<std::string, torch::Tensor> CVAE::forward(const torch::Tensor &x, const torch::Tensor &condition) {
    torch::Tensor z_mean, z_log_std, z;
    torch::Tensor x_recon;
//    std::cout<<"x sizes"<<x.sizes()<<std::endl;
    auto encoder_output = torch::relu(encoder_layer_1->forward(x));
    encoder_output = torch::relu(encoder_layer_2->forward(encoder_output));
    encoder_output = torch::relu(encoder_layer_3->forward(encoder_output));
    z_mean = encoder_mean->forward(encoder_output);
    z_log_std = encoder_log_std->forward(encoder_output);
    z = z_mean + torch::exp(z_log_std) * torch::randn_like(z_log_std);
    auto decoder_input = torch::cat({z, condition}, 1);
//    std::cout << "decoder_input_size: " << decoder_input.sizes() << " " << condition.sizes() << " " << z.sizes()
//              << std::endl;
    auto decoder_output = torch::relu(decoder_layer_1->forward(decoder_input));
//    std::cout << decoder_input << std::endl;
    decoder_output = torch::relu(decoder_layer_2->forward(decoder_output));
    decoder_output = torch::relu(decoder_layer_3->forward(decoder_output));
    std::map<std::string, torch::Tensor> result;
    result["mean"] = z_mean;
    result["log_std"] = z_log_std;
    result["recon"] = decoder_output;
    return result;
}

torch::Tensor CVAE::loss_func(const torch::Tensor &x, const torch::Tensor &recon, const torch::Tensor &mean,
                              const torch::Tensor &log_std) {
    auto recon_loss = torch::sum(torch::pow(x - recon, 2));
    auto kl_loss = 0.5 * torch::sum(torch::exp(log_std) + torch::pow(mean, 2) - 1 - log_std);
    return recon_loss + kl_loss;
}

torch::Tensor CVAE::generate(const torch::Tensor &condition, int latent_size) {
    auto z= torch::randn({1, latent_size});
    auto decoder_input = torch::cat({z, condition}, 1);
    auto decoder_output = torch::relu(decoder_layer_1->forward(decoder_input));
    decoder_output = torch::relu(decoder_layer_2->forward(decoder_output));
    decoder_output = torch::relu(decoder_layer_3->forward(decoder_output));
    return decoder_output.reshape({28, 28});
}
