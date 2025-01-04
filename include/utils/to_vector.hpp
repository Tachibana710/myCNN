#pragma once

#include <eigen3/Eigen/Dense>

#include "datasets/single_data.hpp"
#include "datasets/batch.hpp"

namespace utils
{

// SingleData用
// Width x HeightのMatrix -> Vector
template <typename T, int Width, int Height, int Channel>
inline Eigen::VectorX<T> to_vector(const std::shared_ptr<datasets::SingleData<T, Width, Height, Channel>>& single_data){
    // Eigen::Vector<T, Width*Height*Channel> vec;
    Eigen::VectorX<T> vec(Width*Height*Channel);
    for (int k = 0; k < Channel; ++k){
        for (int i = 0; i < Width; ++i){
            for (int j = 0; j < Height; ++j){
                // vec(i*Height + j) = single_data.data[k](i, j);
                vec(k*Width*Height + i*Height + j) = single_data->data[k](i, j);
            }
        }
    }
    return vec;
}

template <typename T, int Width, int Height, int Channel>
inline Eigen::MatrixX<T> to_2dmatrix(const std::shared_ptr<datasets::SingleData<T, Width, Height, Channel>>& single_data){
    Eigen::MatrixX<T> mat(Width, Height*Channel);
    for (int k = 0; k < Channel; ++k){
        for (int i = 0; i < Width; ++i){
            for (int j = 0; j < Height; ++j){
                mat(i, k*Height + j) = single_data->data[k](i, j);
            }
        }
    }
    return mat;
}

template <typename T>
inline Eigen::VectorX<T> to_vector(const Eigen::MatrixX<T>& mat){
    Eigen::VectorX<T> vec(mat.rows()*mat.cols());
    for (int i = 0; i < mat.rows(); ++i){
        for (int j = 0; j < mat.cols(); ++j){
            vec(i*mat.cols() + j) = mat(i, j);
        }
    }
    return vec;
}

// Batch用
// (Width x Height) x batchsizeのvector<SingleData> -> Vector
// template <typename T, int Width, int Height, int Channel, int BatchSize>
// inline Eigen::Vector<T, Eigen::Dynamic> to_vector(const std::vector<datasets::Batch<T, Width, Height, Channel, BatchSize>>& batch){
//     Eigen::Vector<T,Width*Height*BatchSize> vec;
//     for (int i = 0; i < batch.data.size(); ++i){
//         vec.segment(i*Width*Height, Width*Height) = to_vector(batch.data[i]);
//     }
//     return vec;
// }

} // namespace utils