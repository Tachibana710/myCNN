#pragma once

#include <eigen3/Eigen/Dense>


namespace datasets
{
template <typename T, int Width, int Height, int Channel>
struct SingleData
{
    // Eigen::Matrix<T, Width, Height> data;
    std::array<Eigen::Matrix<T, Width, Height>, Channel> data;
    int label;
    Eigen::MatrixX<T> desired_output;
};
}