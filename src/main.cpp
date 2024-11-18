#include <iostream>

#include <eigen3/Eigen/Dense>

#include "datasets/batch.hpp"

#include "layer/layers.hpp"
#include "utils/to_vector.hpp"

int main(){
    std::string base_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    std::string dataset_path = base_path + "/../datasets/mnist";
    std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    auto batchs = datasets::generate_batches<double, 28, 28, 10>(train_images_path, train_labels_path);

    auto& data = batchs[0].data[0];

    auto flattened_data = utils::to_vector(data);

    layer::AffineLayer<28*28, 10, double> layer;

    layer.forward(flattened_data);

    auto& output = layer.output;

    for (int i = 0; i < 10; ++i){
        std::cout << output(i) << std::endl;
    }
    return 0;
}