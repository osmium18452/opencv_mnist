#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <torch/torch.h>
#include <vector>
#include <opencv2/highgui.hpp>
#include "CVAE.h"
#include "headers/ProgressBar.h"


//convert tensor to opencv mat

cv::Mat tensor2Mat(const torch::Tensor &tensor) {
    int height = tensor.size(0), width = tensor.size(1);
    cv::Mat img(cv::Size(width, height), CV_8U);
    std::memcpy(img.data, tensor.data_ptr(), sizeof(char) * tensor.numel());
    return img;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    int total_epoch = 100;
    int batch_size = 1000;
    int input_size = 28 * 28;
    int condition_size = 10;
    int latent_size = input_size / 8;
    auto dataset = torch::data::datasets::MNIST("../mnist")
            .map(torch::data::transforms::Stack<>());
    std::cout << dataset.size().value() << std::endl;
    int total_iters = dataset.size().value() / batch_size;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     torch::data::DataLoaderOptions().batch_size(batch_size).workers(
                                                             2));
    torch::Device device = torch::Device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "Training on CPU." << std::endl;
    }
    CVAE cvae(input_size, condition_size, latent_size);
    cvae.to(device);
    torch::optim::Adam optimizer(cvae.parameters(), torch::optim::AdamOptions(2e-3));
    for (auto i = 0; i < total_epoch; i++) {
//        std::cout << "epoch " << i << std::endl;
        ProgressBar bar(total_iters);
        bar.set_prefix("epoch: " + std::to_string(i) + "/" + std::to_string(total_epoch) + " ");
        for (torch::data::Example<> &batch: *data_loader) {
            auto data = batch.data.to(device);
            data = torch::reshape(data, {batch_size, -1});
            auto target = batch.target.to(device);
//            std::cout << "data sizes" << data.sizes() << std::endl;
            auto condition = torch::zeros({batch_size, condition_size}).to(torch::kFloat).to(device);
            for (auto j = 0; j < batch_size; j++) {
                condition[j][target[j]] = 1.;
            }
            optimizer.zero_grad();
            auto result = cvae.forward(data, condition);
//            std::cout << "hello" << std::endl;
            auto recon = result["recon"];
            auto mean = result["mean"];
            auto log_std = result["log_std"];
            auto loss = CVAE::loss_func(data, recon, mean, log_std);
            loss.backward();
            optimizer.step();
            bar.set_postfix("loss: " + std::to_string(loss.item<float>()) + " mse loss: " +
                            std::to_string(torch::mean(torch::pow(data - recon, 2)).item<float>()));
            bar.update();
//            std::cout << "epoch: " << i << " loss: " << loss.item<float>() << " mse loss: "
//                      << torch::mean(torch::pow(data - recon, 2)).item() << std::endl;
        }
        bar.close();
    }
    std::vector<cv::Mat> result_list;
    for (auto i = 0; i < 10; i++) {
        auto condition = torch::zeros({1, condition_size}).to(torch::kFloat).to(device);
        condition[0][i] = 1.;
        auto result = cvae.generate(condition,latent_size);
        result_list.push_back(tensor2Mat(result));
    }
    cv::Mat generated_image;
    cv::hconcat(result_list, generated_image);
    cv::imshow("Display window", generated_image);
    cv::waitKey();

//    auto batch = (*data_loader->begin()).data;
    /*cv::Mat final_image;
    for (auto &batch: *data_loader) {
        std::vector<cv::Mat> img_list;
        for (auto i = 0; i < v_pic_num; i++) {
            cv::Mat temp_img;
            std::vector<cv::Mat> img_horizontal_list;
            for (auto j = 0; j < v_pic_num; j++) {
                img_horizontal_list.push_back(
                        tensor2Mat((batch.data[i * v_pic_num + j].squeeze() * 255).to(torch::kU8)));
            }
            cv::hconcat(img_horizontal_list, temp_img);
            img_list.push_back(temp_img);
        }
        cv::vconcat(img_list, final_image);
        cv::imshow("Display window", final_image);
        cv::waitKey();
    }*/
    return 0;
}
