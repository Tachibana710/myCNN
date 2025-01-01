#pragma once

#include <eigen3/Eigen/Dense>

#include <concepts>
#include <type_traits>
#include <memory>
#include <random>

#include "param.hpp"

// class Layer {

// Todo: batch処理に対応させる

namespace layer{

template <int InputDim, int OutputDim, typename T> requires std::is_floating_point_v<T>
class Layer {
public:
    Eigen::Matrix<T, InputDim, 1> input;
    Eigen::Matrix<T, OutputDim, 1> output;
    Eigen::Matrix<T, InputDim, 1> grad;

    virtual void forward(Eigen::Matrix<T, InputDim, 1> signal) = 0;
    virtual void backward(Eigen::Matrix<T, OutputDim, 1> signal) = 0;
};

template <int InputDim, int OutputDim, typename T> requires std::is_floating_point_v<T>
class AffineLayer : public Layer<InputDim, OutputDim, T> {
public:
    std::shared_ptr<Eigen::MatrixX<T>> weights;
    Eigen::Matrix<T, OutputDim, 1> bias;

    Eigen::MatrixX<T> grad_weights;
    int batch_size = 0;

    AffineLayer(){
        weights = std::make_shared<Eigen::MatrixX<T>>(Eigen::Matrix<T, OutputDim, InputDim>::Random());
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<T> dist(0.0, std::sqrt(2.0 / InputDim));
        for (int i = 0; i < OutputDim; ++i){
            for (int j = 0; j < InputDim; ++j){
                (*weights)(i, j) = dist(engine);
            }
        }
        for (int i = 0; i < OutputDim; ++i){
            bias(i) = 0;
        }
        grad_weights = Eigen::MatrixX<T>::Zero(OutputDim, InputDim);
    }

    void forward(Eigen::Matrix<T, InputDim, 1> signal) override {
        this->input = signal;
        this->output = (*weights) * signal + bias;
    }

    void backward(Eigen::Matrix<T, OutputDim, 1> signal) override {
        // Eigen::MatrixX<T> grad_weights = signal * input.transpose();
        grad_weights += static_cast<Eigen::MatrixX<T>>(signal * this->input.transpose());
        batch_size++;
        // (*weights) -= params::learning_rate * static_cast<Eigen::MatrixX<T>>(signal * input.transpose());
        bias -= params::learning_rate * signal;
        this->grad = (*weights).transpose() * signal;
    }

    void update(){
        (*weights) -= params::learning_rate * grad_weights / batch_size;
        grad_weights.setZero();
        batch_size = 0;
    }
        
    
};


template <int Dim, typename T> requires std::is_floating_point_v<T>
class SoftMaxLayer {
public:
    Eigen::Matrix<T, Dim, 1> input;
    Eigen::Matrix<T, Dim, 1> output;
    double loss;

    void forward(Eigen::Matrix<T, Dim, 1> signal){
        this->input = signal;
        T sum = 0;
        T max = input.maxCoeff();
        for (int i = 0; i < Dim; ++i){
            sum += std::exp(input(i)-max);
        }
        for (int i = 0; i < Dim; ++i){
            output(i) = std::exp(input(i)-max) / sum;
        }
    }

    void calc_loss(Eigen::Matrix<T, Dim, 1> label){
        loss = 0;
        double eps = 1e-8;
        for (int i = 0; i < Dim; ++i){
            loss -= label(i) * std::log(output(i) + eps);
        }
    }
};

template <int Dim, typename T> requires std::is_floating_point_v<T>
class ReLULayer {
public:
    Eigen::Matrix<T, Dim, 1> input;
    Eigen::Matrix<T, Dim, 1> output;
    Eigen::Matrix<T, Dim, 1> grad;

    void forward(Eigen::Matrix<T, Dim, 1> input_){
        this->input = input_;
        for (int i = 0; i < Dim; ++i){
            if (input(i) > 0){
                output(i) = input_(i);
            } else {
                // output(i) = 0.01 * input_(i);
                output(i) = 0;
            }
        }
    }

    void backward(Eigen::Matrix<T, Dim, 1> grad_){
        for (int i = 0; i < Dim; ++i){
            if (input(i) > 0){
                this->grad(i) = grad_(i);
            } else {
                // this->grad(i) = 0.01 * grad_(i);
                this->grad(i) = 0;
            }
        }
    }
};

}
