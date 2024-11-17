//load_data.hpp
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

namespace datasets
{

// ヘルパー関数: バイナリデータを読み込む
template<typename T>
T readBigEndian(std::ifstream& stream) {
    T value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        value = (value << 8) | stream.get();
    }
    return value;
}

// MNIST画像データを読み込む
std::vector<Eigen::MatrixXd> loadMNISTImages(const std::string& filePath);

// MNISTラベルデータを読み込む
std::vector<int> loadMNISTLabels(const std::string& filePath);

} // namespace datasets