#include <iostream>

#include <eigen3/Eigen/Dense>

#include "datasets/batch.hpp"
#include "datasets/load_data.hpp"

#include "layer/layers.hpp"
#include "utils/to_vector.hpp"

#include "network/network.hpp"

int main(){

    // load dataset

    std::string base_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    std::string dataset_path = base_path + "/../datasets/mnist";
    std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    // datasets::DataPool<float, 28, 28, 1> data_pool(train_images_path, train_labels_path);

    datasets::DataPool<float, 480, 640, 3> data_pool = datasets::loadJson("/home/taka/mech/jishu_pro/jishupro_ws/src/recognition_pkg/scripts/dataset");

    // create network

    auto my_network = network::Network<float, 480, 640, 3>({
        std::make_shared<layer::AffineLayer<float>>(480*640*3, 50),
        std::make_shared<layer::ReLULayer<float>>(50),
        std::make_shared<layer::AffineLayer<float>>(50, 100),
        std::make_shared<layer::ReLULayer<float>>(100),
        std::make_shared<layer::AffineLayer<float>>(100, 2),
        std::make_shared<layer::ReLULayer<float>>(2),
        std::make_shared<layer::SoftMaxLayer<float>>(2)
    });

    // auto my_network = network::Network<float, 28, 28, 1>("model.json");

    // training

    std::fstream log_file;
    log_file.open("log_loss.csv", std::ios::out);
    log_file << "loss" << std::endl;

    datasets::Batch<float, 480, 640, 3, 10> batch;

    for (int i=0; i < 10000; i++){
        datasets::generate_batch<float, 480, 640, 3, 10>(batch, data_pool);
        // for (auto& dat : batch.data){
        //     dat.desired_output = Eigen::MatrixX<float>::Zero(10, 1);
        //     dat.desired_output(dat.label) = 1;
        // }
        double loss = my_network.train(batch);
        log_file << loss << std::endl;
        std::cout << "batch " << i << " finished.\r" << std::flush;
    }
    log_file.close();
    std::cout << std::endl;

    // my_network.save_model();

    // accuracy check

    int correct = 0;
    datasets::generate_batch<float, 480, 640, 3, 10>(batch, data_pool);
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
        if (max_index == dat->label){
            correct++;
        }
    }
    std::cout << "accuracy: " << (float)correct / 10 << std::endl;

    return 0;
}