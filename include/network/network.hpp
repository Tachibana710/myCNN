#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

#include <nlohmann/json.hpp>

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

    void save_params(){
        std::ofstream output_file("model.json");

        nlohmann::json j;

        for (auto& layer : layers){
            j["structure"].push_back(
                {
                    {"type", layer->name},
                    {"input_dim", layer->input_dim},
                    {"output_dim", layer->output_dim}
                }
            );
        }

        std::function<nlohmann::json(Eigen::MatrixX<T>)> matrix2json = [](Eigen::MatrixX<T> mat){
            nlohmann::json j;
            for (int i = 0; i < mat.rows(); ++i){
                nlohmann::json row;
                for (int j = 0; j < mat.cols(); ++j){
                    row.push_back(mat(i, j));
                }
                j.push_back(row);
            }
            return j;
        };

        for (auto& layer : layers){
            if (typeid(*layer) != typeid(layer::AffineLayer<T>)){
                j["params"].push_back({
                    {"type", layer->name}
                });
                continue;
            }else{
                auto affine_layer = std::dynamic_pointer_cast<layer::AffineLayer<T>>(layer);
                j["params"].push_back(
                    {
                        {"type", layer->name},
                        {"weights", matrix2json(affine_layer->weights)},
                        {"bias", matrix2json(affine_layer->bias)}
                    }
                );
            }
        }

        output_file << j.dump(4) << std::endl;

    }
};

} // namespace network

