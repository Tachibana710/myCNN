#pragma once

#include <eigen3/Eigen/Dense>


namespace datasets
{
template <typename T, int Width, int Height, int Channel>
struct SingleData
{
    // Eigen::Matrix<T, Width, Height> data;
    // std::array<Eigen::Matrix<T, Width, Height>, Channel> data;
    std::array<Eigen::MatrixX<T>, Channel> data;
    int label;
    Eigen::MatrixX<T> desired_output;

    SingleData(){
        for (int i=0; i<Channel; i++){
            data[i] = Eigen::MatrixX<T>::Zero(Width, Height);
        }
    }
};
}