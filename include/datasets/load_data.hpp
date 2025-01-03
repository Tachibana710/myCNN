//load_data.hpp
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <concepts>
#include <type_traits>

#include "datasets/batch.hpp"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <base64/base64.h>

#include <string>


namespace datasets
{

// // ヘルパー関数: バイナリデータを読み込む
// template<typename T>
// T readBigEndian(std::ifstream& stream) {
//     T value = 0;
//     for (size_t i = 0; i < sizeof(T); ++i) {
//         value = (value << 8) | stream.get();
//     }
//     return value;
// }

// // MNIST画像データを読み込む
// template<class T> requires std::is_floating_point_v<T>
// inline std::vector<std::array<Eigen::Matrix<T, 28, 28>,1>> loadMNISTImages(const std::string& filePath) {
//     std::ifstream file(filePath, std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("Failed to open file: " + filePath);
//     }

//     // マジックナンバーとメタデータの読み込み
//     int32_t magicNumber = readBigEndian<int32_t>(file);
//     int32_t numImages = readBigEndian<int32_t>(file);
//     int32_t numRows = readBigEndian<int32_t>(file);
//     int32_t numCols = readBigEndian<int32_t>(file);

//     if (magicNumber != 2051) {
//         throw std::runtime_error("Invalid magic number: " + std::to_string(magicNumber));
//     }

//     // 画像データの読み込み
//     std::vector<std::array<Eigen::Matrix<T, 28, 28>,1>> images(numImages, {{Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(numRows, numCols)}});
//     for (int i = 0; i < numImages; ++i) {
//         for (int r = 0; r < numRows; ++r) {
//             for (int c = 0; c < numCols; ++c) {
//                 images[i][0](r, c) = static_cast<unsigned char>(file.get()) / 255.0; // 正規化
//             }
//         }
//     }
//     return images;
// }

// MNISTラベルデータを読み込む
std::vector<int> loadMNISTLabels(const std::string& filePath);

inline DataPool<int,480,640,3> loadJson(const std::string& dirPath){
    std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)){
        filenames.push_back(entry.path().string());
    }

    DataPool<int,480,640,3> data_pool;

    // 生データをEigen::Matrixに変換
    for (int i = 0; i < filenames.size(); ++i) {
        auto& single_data = data_pool.data[i];
        nlohmann::json jsonData;
        std::ifstream file(filenames[i]);
        // Base64デコード
        std::string decoded_data = base64_decode(jsonData["data"].get<std::string>());

        // デコード結果を生データに変換
        std::vector<uint8_t> raw_data(decoded_data.begin(), decoded_data.end());
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 480; ++row) {
                for (int col = 0; col < 640; ++col) {
                    single_data.data[channel](row, col) = raw_data[(row * 640 + col) * 3 + channel];
                }
            }
        }

        single_data.desired_output = Eigen::MatrixX<int>::Zero(1, 1);
        single_data.desired_output << jsonData["target"][0], jsonData["target"][1];
        
    }

    return data_pool;
}

} // namespace datasets
