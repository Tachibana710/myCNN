//load_data.hpp
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <concepts>
#include <type_traits>

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
template<class T> requires std::is_floating_point_v<T>
std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> loadMNISTImages(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // マジックナンバーとメタデータの読み込み
    int32_t magicNumber = readBigEndian<int32_t>(file);
    int32_t numImages = readBigEndian<int32_t>(file);
    int32_t numRows = readBigEndian<int32_t>(file);
    int32_t numCols = readBigEndian<int32_t>(file);

    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number: " + std::to_string(magicNumber));
    }

    // 画像データの読み込み
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> images(numImages, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(numRows, numCols));
    for (int i = 0; i < numImages; ++i) {
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                images[i](r, c) = static_cast<unsigned char>(file.get()) / 255.0; // 正規化
            }
        }
    }
    return images;
}

// MNISTラベルデータを読み込む
std::vector<int> loadMNISTLabels(const std::string& filePath);

} // namespace datasets