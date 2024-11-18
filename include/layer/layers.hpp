#pragma once

#include <eigen3/Eigen/Dense>

#include <concepts>
#include <type_traits>

// class Layer {

// Todo: batch処理に対応させる

namespace layer{

template <int InputDim, int OutputDim, typename T> requires std::is_floating_point_v<T>
class AffineLayer {
public:
    Eigen::Matrix<T, OutputDim, InputDim+1> weights;
    Eigen::Matrix<T, OutputDim, 1> output;

    AffineLayer(){
        weights = Eigen::Matrix<T, OutputDim, InputDim+1>::Random();
    }

    void forward(Eigen::Matrix<T, InputDim, 1> input){
        Eigen::Matrix<T, InputDim+1, 1> input_bias;
        input_bias << input, 1;
        std::cout << input_bias.size() << std::endl;
        std::cout << weights.size() << std::endl;
        output = weights * input_bias;
    }
};

}
