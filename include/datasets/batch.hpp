#pragma once

#include <vector>
#include "datasets/single_data.hpp"
#include "datasets/load_data.hpp"


namespace datasets
{

template <typename T, int Width, int Height>
struct Batch
{
    std::vector<SingleData<T, Width, Height>> data;
};

template <typename T, int Width, int Height, int BatchNum>
inline std::array<Batch<T, Width, Height>, BatchNum> generate_batches(
    std::string images_path, 
    std::string labels_path)
{
    auto images = std::move(loadMNISTImages<T>(images_path));
    auto labels = std::move(loadMNISTLabels(labels_path));
    int num_images = images.size();
    if (num_images != labels.size()){
        throw std::runtime_error("Number of images and labels do not match");
    }

    std::array<Batch<T, Width, Height>, BatchNum> batchs;

    for (int i = 0; i < num_images; ++i){
        SingleData<T, Width, Height> single_data;
        single_data.data = std::move(images[i]);
        single_data.label = std::move(labels[i]);
        batchs[i % BatchNum].data.push_back(single_data);
    }

    return std::move(batchs);
}

} // namespace datasets