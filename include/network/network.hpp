#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

#include "datasets/single_data.hpp"
#include "layer/layers.hpp"


namespace network{


template <typename T, int Width, int Height> requires std::is_floating_point_v<T>
class Network {
public:
    std::vector<std::shared_ptr<layer::Layer<T>>> layers;

    Network(std::vector<std::shared_ptr<layer::Layer<T>>> layers_){
        layers = layers_;
    }

    Eigen::VectorX<T> forward(datasets::SingleData<T, Width, Height> data){
        auto flattened_data = utils::to_vector(data);
        layers[0]->forward(flattened_data);
        for (int i = 1; i < layers.size(); ++i){
            layers[i]->forward(layers[i-1]->output);
        }
        return layers.back()->output;
    }

    void backward(Eigen::VectorX<T> grad){
        layers.back()->backward(grad);
        for (int i = layers.size()-2; i >= 0; --i){
            layers[i]->backward(layers[i+1]->grad);
        }
    }

    template <int BatchSize>
    double train(datasets::Batch<T, Width, Height, BatchSize> batch){
        double loss_sum = 0;
        for (auto& dat : batch.data){
            auto output = forward(dat);

            std::shared_ptr<layer::OutputLayer<T>> output_layer = std::dynamic_pointer_cast<layer::OutputLayer<T>>(layers.back());
            output_layer->calc_loss(dat.desired_output);
            loss_sum += output_layer->loss;

            backward(dat.desired_output);
        }
        for (auto& layer : layers){
            layer->update();
        }
        return loss_sum / batch.data.size();
    }

    Eigen::VectorX<T> predict(datasets::SingleData<T, Width, Height> data){
        auto flattened_data = utils::to_vector(data);
        layers[0]->forward(flattened_data);
        for (int i = 1; i < layers.size(); ++i){
            layers[i]->forward(layers[i-1]->output);
        }
        return layers.back()->output;
    }
};

} // namespace network

