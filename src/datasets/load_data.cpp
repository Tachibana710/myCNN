#include "datasets/load_data.hpp"


namespace datasets
{

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