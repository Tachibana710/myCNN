#pragma once

#include <vector>
#include "datasets/single_data.hpp"
#include "datasets/load_data.hpp"


namespace datasets
{

template <typename T, int Width, int Height>
class Batch
{
public:
    std::vector<SingleData<T, Width, Height>> data;

    void load_dataset(std::string images_path, std::string labels_path){
        auto images = std::move(datasets::loadMNISTImages(images_path));
        auto labels = std::move(datasets::loadMNISTLabels(labels_path));
        int num_images = images.size();
        if (num_images != labels.size()){
            throw std::runtime_error("Number of images and labels do not match");
        }

        for (int i = 0; i < num_images; ++i){
            SingleData<T, Width, Height> single_data;
            single_data.data = std::move(images[i]);
            single_data.label = std::move(labels[i]);
            data.push_back(single_data);
        }
    }
};

} // namespace datasets