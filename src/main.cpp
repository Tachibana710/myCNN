#include <iostream>

#include <eigen3/Eigen/Dense>

#include "datasets/batch.hpp"

int main(){

    datasets::Batch<double, 28, 28> batch;

    std::string base_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    std::string dataset_path = base_path + "/../datasets/mnist";
    std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    auto batchs = datasets::generate_batches<double, 28, 28, 10>(train_images_path, train_labels_path);

    for (int i = 0; i < 28; ++i){
        for (int j = 0; j < 28; ++j){
            if (batchs[0].data[0].data(i, j) > 0.5){
                std::cout << "■";
            } else {
                std::cout << "□";
            }
            // std::cout << batch.data[0].data(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "label: " << batchs[0].data[0].label << std::endl;
    return 0;
}