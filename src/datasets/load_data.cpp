#include "datasets/load_data.hpp"

namespace datasets
{

// MNIST画像データを読み込む
std::vector<Eigen::MatrixXd> loadMNISTImages(const std::string& filePath) {
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
    std::vector<Eigen::MatrixXd> images(numImages, Eigen::MatrixXd(numRows, numCols));
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
std::vector<int> loadMNISTLabels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // マジックナンバーとメタデータの読み込み
    int32_t magicNumber = readBigEndian<int32_t>(file);
    int32_t numLabels = readBigEndian<int32_t>(file);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number: " + std::to_string(magicNumber));
    }

    // ラベルデータの読み込み
    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        labels[i] = static_cast<unsigned char>(file.get());
    }
    return labels;
}

} // namespace datasets