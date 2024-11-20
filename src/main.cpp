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

    datasets::DataPool<float, 28, 28> data_pool(train_images_path, train_labels_path);

    // auto batchs = datasets::generate_batches<float, 28, 28, 10>(train_images_path, train_labels_path);


    layer::AffineLayer<28*28, 50, float> affine_layer1;
    layer::ReLULayer<50, float> relu_layer1;
    layer::AffineLayer<50, 100, float> affine_layer2;
    layer::ReLULayer<100, float> relu_layer2;
    layer::AffineLayer<100, 10, float> affine_layer3;
    layer::ReLULayer<10, float> relu_layer3;
    layer::SoftMaxLayer<10, float> softmax_layer;

    std::fstream log_file;
    log_file.open("log_loss.csv", std::ios::out);
    log_file << "loss" << std::endl;

    datasets::Batch<float, 28, 28, 100> batch;

    for (int i=0; i < 10000; i++){
        datasets::generate_batch<float, 28, 28, 100>(batch, data_pool);
        double softmax_loss_sum = 0;
        for (auto& dat : batch.data){
            auto flattened_data = utils::to_vector(dat);
            affine_layer1.forward(flattened_data);
            relu_layer1.forward(affine_layer1.output);
            affine_layer2.forward(relu_layer1.output);
            relu_layer2.forward(affine_layer2.output);
            affine_layer3.forward(relu_layer2.output);
            relu_layer3.forward(affine_layer3.output);
            softmax_layer.forward(relu_layer3.output);

            Eigen::Matrix<float, 10, 1> grad;
            grad << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
            grad(dat.label) = 1;
            softmax_layer.calc_loss(grad);
            softmax_loss_sum += softmax_layer.loss;
            grad = softmax_layer.output - grad;
            relu_layer3.backward(grad);
            affine_layer3.backward(relu_layer3.grad);
            relu_layer2.backward(affine_layer3.grad);
            affine_layer2.backward(relu_layer2.grad);
            relu_layer1.backward(affine_layer2.grad);
            affine_layer1.backward(relu_layer1.grad);
            static int count = 0;
            std::cout << "count:" << count++ << "\r";
        }
        affine_layer1.update();
        affine_layer2.update();
        affine_layer3.update();
        log_file << softmax_loss_sum / batch.data.size() << std::endl;
        std::cout << "batch" << i << " finished" << std::endl;
    }

    int correct = 0;
    datasets::generate_batch<float, 28, 28, 100>(batch, data_pool);
    for (auto& dat : batch.data){
        auto flattened_data = utils::to_vector(dat);
        affine_layer1.forward(flattened_data);
        relu_layer1.forward(affine_layer1.output);
        affine_layer2.forward(relu_layer1.output);
        relu_layer2.forward(affine_layer2.output);
        affine_layer3.forward(relu_layer2.output);
        relu_layer3.forward(affine_layer3.output);
        softmax_layer.forward(relu_layer3.output);

        int max_index = -INFINITY;
        float max_value = 0;
        for (int i = 0; i < 10; ++i){
            if (softmax_layer.output(i) > max_value){
                max_value = softmax_layer.output(i);
                max_index = i;
            }
        }
        if (max_index == dat.label){
            correct++;
        }
    }
    std::cout << "accuracy: " << (float)correct / 100 << std::endl;


    // auto& data = batchs[0].data[0];

    // auto flattened_data = utils::to_vector(data);
    // for (auto& val : flattened_data){
    //     val -= 0.5;
    // }
    // affine_layer1.forward(flattened_data);
    // std::cout << "affine_layer1.output" << std::endl;
    // std::cout << affine_layer1.output << std::endl;
    // relu_layer1.forward(affine_layer1.output);
    // std::cout << "relu_layer1.output" << std::endl;
    // std::cout << relu_layer1.output << std::endl;
    // affine_layer2.forward(relu_layer1.output);
    // std::cout << "affine_layer2.output" << std::endl;
    // std::cout << affine_layer2.output << std::endl;
    // relu_layer2.forward(affine_layer2.output);
    // std::cout << "relu_layer2.output" << std::endl;
    // std::cout << relu_layer2.output << std::endl;
    // affine_layer3.forward(relu_layer2.output);
    // std::cout << "affine_layer3.output" << std::endl;
    // std::cout << affine_layer3.output << std::endl;
    // relu_layer3.forward(affine_layer3.output);
    // std::cout << "relu_layer3.output" << std::endl;
    // std::cout << relu_layer3.output << std::endl;
    // softmax_layer.forward(relu_layer3.output);
    // std::cout << "softmax_layer.output" << std::endl;
    // std::cout << softmax_layer.output << std::endl;

    // // return 0;

    // std::cout << "start training" << std::endl;

    // int correct = 0;
    // for (auto& dat : batchs[0].data){
    //     auto flattened_data = utils::to_vector(dat);
    //     affine_layer1.forward(flattened_data);
    //     relu_layer1.forward(affine_layer1.output);
    //     affine_layer2.forward(relu_layer1.output);
    //     relu_layer2.forward(affine_layer2.output);
    //     affine_layer3.forward(relu_layer2.output);
    //     relu_layer3.forward(affine_layer3.output);

    //     int max_index = 0;
    //     float max_value = 0;
    //     for (int i = 0; i < 10; ++i){
    //         if (relu_layer3.output(i) > max_value){
    //             max_value = softmax_layer.output(i);
    //             max_index = i;
    //         }
    //     }
    //     if (max_index == dat.label){
    //         correct++;
    //     }
    // }
    // std::cout << "accuracy: " << (float)correct / batchs[0].data.size() << std::endl;
    
    // std::fstream log_file;
    // log_file.open("log_loss.csv", std::ios::out);

    // double softmax_loss_sum = 0;
    // for (int i = 0; i < 500; i++){
    //     for (auto& dat : batchs[0].data){
    //         auto flattened_data = utils::to_vector(dat);
    //         affine_layer1.forward(flattened_data);
    //         relu_layer1.forward(affine_layer1.output);
    //         affine_layer2.forward(relu_layer1.output);
    //         relu_layer2.forward(affine_layer2.output);
    //         affine_layer3.forward(relu_layer2.output);
    //         relu_layer3.forward(affine_layer3.output);
    //         softmax_layer.forward(relu_layer3.output);

    //         Eigen::Matrix<float, 10, 1> grad;
    //         grad << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //         grad(dat.label) = 1;
    //         softmax_layer.calc_loss(grad);
    //         softmax_loss_sum += softmax_layer.loss;
    //         grad = softmax_layer.output - grad;
    //         relu_layer3.backward(grad);
    //         affine_layer3.backward(relu_layer3.grad);
    //         relu_layer2.backward(affine_layer3.grad);
    //         affine_layer2.backward(relu_layer2.grad);
    //         relu_layer1.backward(affine_layer2.grad);
    //         affine_layer1.backward(relu_layer1.grad);
    //         static int count = 0;
    //         std::cout << "count:" << count++ << "\r";
    //     }
    //     affine_layer1.update();
    //     affine_layer2.update();
    //     affine_layer3.update();
    //     log_file << softmax_loss_sum / batchs[i % 10].data.size() << std::endl;
    //     softmax_loss_sum = 0;
    //     std::cout << "batch" << i << " finished" << std::endl;
    // }

    // flattened_data = utils::to_vector(data);
    // for (auto& val : flattened_data){
    //     val -= 0.5;
    // }
    // std::cout << "input" << std::endl;
    // std::cout << flattened_data.transpose() << std::endl;
    // affine_layer1.forward(flattened_data);
    // std::cout << "affine_layer1.output" << std::endl;
    // std::cout << affine_layer1.output.transpose() << std::endl;
    // relu_layer1.forward(affine_layer1.output);
    // std::cout << "relu_layer1.output" << std::endl;
    // std::cout << relu_layer1.output.transpose() << std::endl;
    // affine_layer2.forward(relu_layer1.output);
    // std::cout << "affine_layer2.output" << std::endl;
    // std::cout << affine_layer2.output.transpose() << std::endl;
    // relu_layer2.forward(affine_layer2.output);
    // std::cout << "relu_layer2.output" << std::endl;
    // std::cout << relu_layer2.output.transpose() << std::endl;
    // affine_layer3.forward(relu_layer2.output);
    // std::cout << "affine_layer3.output" << std::endl;
    // std::cout << affine_layer3.output.transpose() << std::endl;
    // relu_layer3.forward(affine_layer3.output);
    // std::cout << "relu_layer3.output" << std::endl;
    // std::cout << relu_layer3.output.transpose() << std::endl;
    // softmax_layer.forward(relu_layer3.output);
    // std::cout << "softmax_layer.output" << std::endl;
    // std::cout << softmax_layer.output.transpose() << std::endl;

    // // return 0;


    // correct = 0;
    // for (auto& dat : batchs[0].data){
    //     auto flattened_data = utils::to_vector(dat);
    //     affine_layer1.forward(flattened_data);
    //     relu_layer1.forward(affine_layer1.output);
    //     affine_layer2.forward(relu_layer1.output);
    //     relu_layer2.forward(affine_layer2.output);
    //     affine_layer3.forward(relu_layer2.output);
    //     relu_layer3.forward(affine_layer3.output);
    //     softmax_layer.forward(relu_layer3.output);

    //     // std::cout << "label: " << dat.label << std::endl;
    //     // std::cout << std::endl;

    //     int max_index = 0;
    //     float max_value = 0;
    //     for (int i = 0; i < 10; ++i){
    //         if (relu_layer3.output(i) > max_value){
    //             max_value = softmax_layer.output(i);
    //             max_index = i;
    //         }
    //     }
    //     if (max_index == dat.label){
    //         correct++;
    //     }
    // }
    // std::cout << "accuracy: " << (float)correct / batchs[0].data.size() << std::endl;

    // layer.forward(flattened_data);

    // auto& output = layer.output;

    // for (int i = 0; i < 10; ++i){
    //     std::cout << output(i) << std::endl;
    // }
    return 0;
}