#pragma once

#include <eigen3/Eigen/Dense>

#include <concepts>
#include <type_traits>
#include <memory>

#include "param.hpp"

// class Layer {

// Todo: batch処理に対応させる

namespace layer{

template <int InputDim, int OutputDim, typename T> requires std::is_floating_point_v<T>
class AffineLayer {
public:
    std::shared_ptr<Eigen::MatrixX<T>> weights;
    Eigen::Matrix<T, OutputDim, 1> bias;
    Eigen::Matrix<T, InputDim, 1> input;
    Eigen::Matrix<T, InputDim, 1> grad;
    Eigen::Matrix<T, OutputDim, 1> output;

    AffineLayer(){
        weights = std::make_shared<Eigen::MatrixX<T>>(Eigen::Matrix<T, OutputDim, InputDim>::Random());
        bias = Eigen::Matrix<T, OutputDim, 1>::Random();
    }

    void forward(Eigen::Matrix<T, InputDim, 1> input_){
        this->input = input_;
        output = (*weights) * input_ + bias;
    }

    void backward(Eigen::Matrix<T, OutputDim, 1> signal){
        // Eigen::MatrixX<T> grad_weights = signal * input.transpose();
        (*weights) -= params::learning_rate * static_cast<Eigen::MatrixX<T>>(signal * input.transpose());
        bias -= params::learning_rate * signal;
        this->grad = (*weights).transpose() * signal;
    }
        
};


template <int Dim, typename T> requires std::is_floating_point_v<T>
class SoftMaxLayer {
public:
    Eigen::Matrix<T, Dim, 1> input;
    Eigen::Matrix<T, Dim, 1> output;

    void forward(Eigen::Matrix<T, Dim, 1> input){
        this->input = input;
        T sum = 0;
        for (int i = 0; i < Dim; ++i){
            sum += std::exp(input(i));
        }
        for (int i = 0; i < Dim; ++i){
            output(i) = std::exp(input(i)) / sum;
        }
    }
};

template <int Dim, typename T> requires std::is_floating_point_v<T>
class ReLULayer {
public:
    Eigen::Matrix<T, Dim, 1> input;
    Eigen::Matrix<T, Dim, 1> output;
    Eigen::Matrix<T, Dim, 1> grad;

    void forward(Eigen::Matrix<T, Dim, 1> input){
        this->input = input;
        for (int i = 0; i < Dim; ++i){
            if (input(i) > 0){
                output(i) = input(i);
            } else {
                output(i) = 0;
            }
        }
    }

    void backward(Eigen::Matrix<T, Dim, 1> grad){
        for (int i = 0; i < Dim; ++i){
            if (input(i) > 0){
                this->grad(i) = grad(i);
            } else {
                this->grad(i) = 0;
            }
        }
    }
};

}
