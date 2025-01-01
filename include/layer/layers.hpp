#pragma once

#include <eigen3/Eigen/Dense>

#include <concepts>
#include <type_traits>
#include <memory>
#include <random>

#include "param.hpp"

namespace layer{

template <typename T> requires std::is_floating_point_v<T>
class Layer {
public:

    std::string name;

    Eigen::MatrixX<T> input;
    Eigen::MatrixX<T> output;
    Eigen::MatrixX<T> grad;

    const int input_dim;
    const int output_dim;

    Layer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim){
        input = Eigen::MatrixX<T>::Zero(input_dim, 1);
        output = Eigen::MatrixX<T>::Zero(output_dim, 1);
        grad = Eigen::MatrixX<T>::Zero(input_dim, 1);
    }
    virtual void forward(Eigen::MatrixX<T> signal) = 0;
    virtual void backward(Eigen::MatrixX<T> signal) = 0;
    virtual void update(){};
};

template <typename T> requires std::is_floating_point_v<T>
class AffineLayer : public Layer<T> {
public:

    Eigen::MatrixX<T> weights;
    Eigen::MatrixX<T> bias;

    Eigen::MatrixX<T> grad_weights;
    int batch_size = 0;

    AffineLayer(int input_dim, int output_dim) : Layer<T>(input_dim, output_dim){
        this->name = "AffineLayer";

        weights = Eigen::MatrixX<T>::Zero(output_dim, input_dim);
        bias = Eigen::MatrixX<T>::Zero(output_dim, 1);

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<T> dist(0.0, std::sqrt(2.0 / this->input_dim));
        for (int i = 0; i < this->output_dim; ++i){
            for (int j = 0; j < this->input_dim; ++j){
                weights(i, j) = dist(engine);
            }
        }
        for (int i = 0; i < output_dim; ++i){
            bias(i) = 0;
        }
        grad_weights = Eigen::MatrixX<T>::Zero(this->output_dim, this->input_dim);
    }

    void forward(Eigen::MatrixX<T> signal) override {
        this->input = signal;
        this->output = weights * signal + bias;
    }

    void backward(Eigen::MatrixX<T> signal) override {
        grad_weights += static_cast<Eigen::MatrixX<T>>(signal * this->input.transpose());
        batch_size++;
        bias -= params::learning_rate * signal;
        this->grad = weights.transpose() * signal;
    }

    void update() override {
        weights -= params::learning_rate * grad_weights / batch_size;
        grad_weights.setZero();
        batch_size = 0;
    }
         
};

template <typename T> requires std::is_floating_point_v<T>
class ReLULayer : public Layer<T> {
public:

    const int dim;

    ReLULayer(int input_dim) : Layer<T>(input_dim, input_dim), dim(input_dim){
        this->name = "ReLULayer";
    }

    void forward(Eigen::MatrixX<T> input_) override {
        this->input = input_;
        for (int i = 0; i < dim; ++i){
            if (this->input(i) > 0){
                this->output(i) = input_(i);
            } else {
                this->output(i) = 0;
            }
        }
    }

    void backward(Eigen::MatrixX<T> grad_) override {
        for (int i = 0; i < dim; ++i){
            if (this->input(i) > 0){
                this->grad(i) = grad_(i);
            } else {
                this->grad(i) = 0;
            }
        }
    }
};

template <typename T> requires std::is_floating_point_v<T>
class OutputLayer : public Layer<T> {
public:
    OutputLayer(int input_dim, int output_dim) : Layer<T>(input_dim, output_dim){}
    double loss;
    virtual void calc_loss(Eigen::MatrixX<T> desired_output) = 0;
};


template <typename T> requires std::is_floating_point_v<T>
class SoftMaxLayer : public OutputLayer<T> {
public:

    const int dim;

    SoftMaxLayer(int input_dim) : OutputLayer<T>(input_dim, input_dim), dim(input_dim){
        this->name = "SoftMaxLayer";
    }


    void forward(Eigen::MatrixX<T> signal) override {
        this->input = signal;
        T sum = 0;
        T max = this->input.maxCoeff();
        for (int i = 0; i < dim; ++i){
            sum += std::exp(this->input(i)-max);
        }
        for (int i = 0; i < dim; ++i){
            this->output(i) = std::exp(this->input(i)-max) / sum;
        }
    }

    void backward(Eigen::MatrixX<T> grad) override {
        this->grad = this->output - grad;
    }

    void calc_loss(Eigen::MatrixX<T> desired_output) override {
        this->loss = 0;
        double eps = 1e-8;
        for (int i = 0; i < dim; ++i){
            this->loss -= desired_output(i) * std::log(this->output(i) + eps);
        }
    }
};

}
