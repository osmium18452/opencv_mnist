#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <torch/torch.h>
#include <vector>


//convert tensor to opencv mat

cv::Mat tensor2Mat(const torch::Tensor &tensor) {
    int height = tensor.size(0), width = tensor.size(1);
    cv::Mat img(cv::Size(width, height), CV_8U);
    std::memcpy(img.data, tensor.data_ptr(), sizeof(char) * tensor.numel());
    return img;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    int batch_size = 32;
    torch::Device device(torch::kCUDA, 1);
    auto dataset = torch::data::datasets::MNIST("../mnist")
            .map(torch::data::transforms::Stack<>());
    std::cout << dataset.size().value() << std::endl;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     torch::data::DataLoaderOptions().batch_size(batch_size).workers(
                                                             2));
    auto batch = (*data_loader->begin()).data;
    cv::Mat final_image;
    std::vector<cv::Mat> img_list;
    for (auto i = 0; i < batch_size; i++) {
        auto tensor_image = (batch[i].squeeze() * 255).to(torch::kU8);
        img_list.push_back(tensor2Mat(tensor_image));
    }
    cv::hconcat(img_list, final_image);
    cv::imshow("Display window", final_image);
    cv::waitKey();
    return 0;
}
