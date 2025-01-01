#include <iostream>

#include <eigen3/Eigen/Dense>

#include "datasets/batch.hpp"

#include "layer/layers.hpp"
#include "utils/to_vector.hpp"

#include "network/network.hpp"

int main(){

    // load dataset

    std::string base_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    std::string dataset_path = base_path + "/../datasets/mnist";
    std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    datasets::DataPool<float, 28, 28> data_pool(train_images_path, train_labels_path);

    // create network

    auto my_network = network::Network<float, 28, 28>({
        std::make_shared<layer::AffineLayer<float>>(28*28, 50),
        std::make_shared<layer::ReLULayer<float>>(50),
        std::make_shared<layer::AffineLayer<float>>(50, 100),
        std::make_shared<layer::ReLULayer<float>>(100),
        std::make_shared<layer::AffineLayer<float>>(100, 10),
        std::make_shared<layer::ReLULayer<float>>(10),
        std::make_shared<layer::SoftMaxLayer<float>>(10)
    });

    // training

    std::fstream log_file;
    log_file.open("log_loss.csv", std::ios::out);
    log_file << "loss" << std::endl;

    datasets::Batch<float, 28, 28, 100> batch;

    for (int i=0; i < 10000; i++){
        datasets::generate_batch<float, 28, 28, 100>(batch, data_pool);
        for (auto& dat : batch.data){
            dat.desired_output = Eigen::MatrixX<float>::Zero(10, 1);
            dat.desired_output(dat.label) = 1;
        }
        double loss = my_network.train(batch);
        log_file << loss << std::endl;
        std::cout << "batch " << i << " finished.\r" << std::flush;
    }
    log_file.close();
    std::cout << std::endl;

    my_network.save_params();

    // accuracy check

    int correct = 0;
    datasets::generate_batch<float, 28, 28, 100>(batch, data_pool);
    for (auto& dat : batch.data){
        auto output = my_network.predict(dat);
        int max_index = 0;
        float max_value = -INFINITY;
        for (int i = 0; i < 10; ++i){
            if (output(i) > max_value){
                max_value = output(i);
                max_index = i;
            }
        }
        if (max_index == dat.label){
            correct++;
        }
    }
    std::cout << "accuracy: " << (float)correct / 100 << std::endl;

    return 0;
}