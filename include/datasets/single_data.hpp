#pragma once

#include <eigen3/Eigen/Dense>


namespace datasets
{
template <typename T, int Width, int Height>
struct SingleData
{
    Eigen::Matrix<T, Width, Height> data;
    int label;
};
}