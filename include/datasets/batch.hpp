#pragma once

#include <vector>
#include <random>


#include "datasets/single_data.hpp"
#include "datasets/load_data.hpp"


namespace datasets
{

template <typename T, int Width, int Height, int Channel, int BatchSize>
struct Batch
{
    // std::vector<SingleData<T, Width, Height>> data;
    std::array<SingleData<T, Width, Height, Channel>, BatchSize> data;

    void add_noise(double noise_ratio){
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_real_distribution<float> noise_dist(0, 1);

    for (auto& dat : data.data){
        for (int i = 0; i < 28; ++i){
            for (int j = 0; j < 28; ++j){
                if (noise_dist(engine) < noise_ratio){
                    dat.data(i, j) = noise_dist(engine);
                }
            }
        }
    }
}
};

template <typename T, int Width, int Height, int Channel>
struct DataPool
{
    std::vector<SingleData<T, Width, Height, Channel>> data;


    DataPool(std::string images_path, std::string labels_path)
    {
        auto images = std::move(loadMNISTImages<T>(images_path));
        auto labels = std::move(loadMNISTLabels(labels_path));
        int num_images = images.size();
        if (num_images != labels.size())
        {
            throw std::runtime_error("Number of images and labels do not match");
        }

        for (int i = 0; i < num_images; ++i)
        {
            SingleData<T, Width, Height, 1> single_data;
            single_data.data = std::move(images[i]);
            single_data.label = std::move(labels[i]);
            data.push_back(single_data);
        }
    }

    void add_noise(double noise_ratio){
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::uniform_real_distribution<float> noise_dist(0, 1);

        for (auto& dat : data){
            for (int i = 0; i < 28; ++i){
                for (int j = 0; j < 28; ++j){
                    if (noise_dist(engine) < noise_ratio){
                        dat.data[0](i, j) = noise_dist(engine);
                    }
                }
            }
        }
    }
};

// template <typename T, int Width, int Height, int BatchNum>
// inline std::array<Batch<T, Width, Height>, BatchNum> generate_batches(
//     std::string images_path, 
//     std::string labels_path)
// {
//     auto images = std::move(loadMNISTImages<T>(images_path));
//     auto labels = std::move(loadMNISTLabels(labels_path));
//     int num_images = images.size();
//     if (num_images != labels.size()){
//         throw std::runtime_error("Number of images and labels do not match");
//     }

//     std::array<Batch<T, Width, Height>, BatchNum> batchs;

//     for (int i = 0; i < num_images; ++i){
//         SingleData<T, Width, Height> single_data;
//         single_data.data = std::move(images[i]);
//         single_data.label = std::move(labels[i]);
//         batchs[i % BatchNum].data.push_back(single_data);
//     }

//     return std::move(batchs);
// }

template <typename T, int Width, int Height, int Channel, int BatchSize>
inline void generate_batch(
    Batch<T, Width, Height, Channel, BatchSize>& batch,
    DataPool<T, Width, Height, Channel>& data_pool)
{
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_int_distribution<int> dist(0, data_pool.data.size());

    for (int i = 0; i < BatchSize; ++i)
    {
        batch.data[i] = data_pool.data[dist(engine)];
    }
}

} // namespace datasets