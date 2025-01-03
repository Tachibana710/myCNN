#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#include "datasets/single_data.hpp"
#include "layer/layers.hpp"


namespace network{


template <typename T, int Width, int Height, int Channel> requires std::is_floating_point_v<T>
class Network {
public:
    std::vector<std::shared_ptr<layer::Layer<T>>> layers;

    Network(std::vector<std::shared_ptr<layer::Layer<T>>> layers_){
        layers = layers_;
    }

    Network(std::string json_path){
        this->load_model(json_path);
    }

    Eigen::VectorX<T> forward(datasets::SingleData<T, Width, Height, Channel> data){
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
    double train(datasets::Batch<T, Width, Height, Channel, BatchSize> batch){
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

    Eigen::VectorX<T> predict(datasets::SingleData<T, Width, Height, Channel> data){
        auto flattened_data = utils::to_vector(data);
        layers[0]->forward(flattened_data);
        for (int i = 1; i < layers.size(); ++i){
            layers[i]->forward(layers[i-1]->output);
        }
        return layers.back()->output;
    }

    void save_model(){
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

    void load_model(std::string json_path){
        std::ifstream input_file(json_path);
        nlohmann::json json;
        input_file >> json;

        layers.clear();
        for (auto& layer : json["structure"]){
            if (layer["type"] == "AffineLayer"){
                layers.push_back(std::make_shared<layer::AffineLayer<T>>(layer["input_dim"], layer["output_dim"]));
            }else if (layer["type"] == "ReLULayer"){
                layers.push_back(std::make_shared<layer::ReLULayer<T>>(layer["input_dim"]));
            }else if (layer["type"] == "SoftMaxLayer"){
                layers.push_back(std::make_shared<layer::SoftMaxLayer<T>>(layer["input_dim"]));
            }
        }

        for (int i = 0; i < layers.size(); ++i){
            if (json["params"][i]["type"] != "AffineLayer"){
                continue;
            }
            auto affine_layer = std::dynamic_pointer_cast<layer::AffineLayer<T>>(layers[i]);
            for (int j = 0; j < affine_layer->weights.rows(); ++j){
                for (int k = 0; k < affine_layer->weights.cols(); ++k){
                    affine_layer->weights(j, k) = json["params"][i]["weights"][j][k];
                }
            }
            for (int j = 0; j < affine_layer->bias.rows(); ++j){
                for (int k = 0; k < affine_layer->bias.cols(); ++k){
                    affine_layer->bias(j, k) = json["params"][i]["bias"][j][k];
                }
            }
        }
    }
};

} // namespace network

