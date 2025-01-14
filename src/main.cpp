#include <iostream>

#include <eigen3/Eigen/Dense>

#include "datasets/batch.hpp"

#include "layer/layers.hpp"
#include "utils/to_vector.hpp"

#include "network/network.hpp"

#include <chrono>
#include <cstdlib>

int main(){

    // load dataset

    std::string base_path = std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of("/\\"));
    std::string dataset_path = base_path + "/../datasets/mnist";
    std::string train_images_path = dataset_path + "/train-images.idx3-ubyte";
    std::string train_labels_path = dataset_path + "/train-labels.idx1-ubyte";

    datasets::DataPool<float, 28, 28, 1> data_pool(train_images_path, train_labels_path);
    std::cout << data_pool.data.size() << " images loaded." << std::endl;

    // create network
    datasets::Batch<float, 28, 28, 1, 100> batch;

    network::Network<float, 28, 28, 1> my_network;

    if (std::getenv("LOAD_JSON") != nullptr){
        my_network = network::Network<float, 28, 28, 1>("model.json");
    } else {
        my_network = network::Network<float, 28, 28, 1>({
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


        double now_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        for (int i=0; i < 10000; i++){
            datasets::generate_batch<float, 28, 28, 1, 100>(batch, data_pool);
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

        double end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << "training time: " << (end_time - now_time) / 1000.0 << "s" << std::endl;

        if (std::getenv("SAVE_JSON") != nullptr){
            my_network.save_model();
        }
    }

    // accuracy check

    int correct = 0;
    int cnt = 0;

    datasets::Batch<float, 28, 28, 1, 1000> test_batch;

    datasets::generate_batch<float, 28, 28, 1, 1000>(test_batch, data_pool);
    for (auto& dat : test_batch.data){
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
        cnt++;
    }
    std::cout << "accuracy: " << (float)correct / cnt << std::endl;
    std::cout << "correct: " << correct << " / " << cnt << std::endl;

    return 0;
}