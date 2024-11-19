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

    auto batchs = datasets::generate_batches<float, 28, 28, 10>(train_images_path, train_labels_path);

    auto& data = batchs[0].data[0];

    auto flattened_data = utils::to_vector(data);

    layer::AffineLayer<28*28, 100, float> affine_layer1;
    layer::ReLULayer<100, float> relu_layer1;
    layer::AffineLayer<100, 100, float> affine_layer2;
    layer::ReLULayer<100, float> relu_layer2;
    layer::AffineLayer<100, 10, float> affine_layer3;
    layer::ReLULayer<10, float> relu_layer3;
    layer::SoftMaxLayer<10, float> softmax_layer;

    std::cout << "start training" << std::endl;

    int correct = 0;
    for (auto& dat : batchs[0].data){
        auto flattened_data = utils::to_vector(dat);
        affine_layer1.forward(flattened_data);
        relu_layer1.forward(affine_layer1.output);
        affine_layer2.forward(relu_layer1.output);
        relu_layer2.forward(affine_layer2.output);
        affine_layer3.forward(relu_layer2.output);
        relu_layer3.forward(affine_layer3.output);

        int max_index = 0;
        float max_value = 0;
        for (int i = 0; i < 10; ++i){
            if (relu_layer3.output(i) > max_value){
                max_value = softmax_layer.output(i);
                max_index = i;
            }
        }
        if (max_index == dat.label){
            correct++;
        }
    }
    std::cout << "accuracy: " << (float)correct / batchs[0].data.size() << std::endl;
    
    for (int i = 0; i < 10; i++){
        for (auto& dat : batchs[0].data){
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
            grad(data.label) = 1;
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
        std::cout << "batch" << i << " finished" << std::endl;
    }


    correct = 0;
    for (auto& dat : batchs[0].data){
        auto flattened_data = utils::to_vector(dat);
        affine_layer1.forward(flattened_data);
        relu_layer1.forward(affine_layer1.output);
        affine_layer2.forward(relu_layer1.output);
        relu_layer2.forward(affine_layer2.output);
        affine_layer3.forward(relu_layer2.output);
        relu_layer3.forward(affine_layer3.output);

        int max_index = 0;
        float max_value = 0;
        for (int i = 0; i < 10; ++i){
            if (relu_layer3.output(i) > max_value){
                max_value = softmax_layer.output(i);
                max_index = i;
            }
        }
        if (max_index == dat.label){
            correct++;
        }
    }
    std::cout << "accuracy: " << (float)correct / batchs[0].data.size() << std::endl;

    // layer.forward(flattened_data);

    // auto& output = layer.output;

    // for (int i = 0; i < 10; ++i){
    //     std::cout << output(i) << std::endl;
    // }
    return 0;
}