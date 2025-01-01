namespace network{


template <typename T, int Width, int Height> requires std::is_floating_point_v<T>
class Network {
public:
    std::vector<std::shared_ptr<layer::LayerBase>> layers;

    Network(std::vector<std::shared_ptr<layer::LayerBase>> layers_){
        layers = layers_;
    }

    std::vector<T> forward(datasets::SingleData<T, Width, Height> data){
        std::vector<T> flattened_data = utils::to_vector(data.data);
        layers[0]->forward(flattened_data);
        for (int i = 1; i < layers.size(); ++i){
            layers[i]->forward(layers[i-1]->output);
        }
        return layers.back()->output;
    }

    void backward(Eigen::VectorX<T> grad){
        for (int i = layers.size()-1; i >= 0; --i){
            layers[i]->backward(grad);
        }
        // for (auto layer : layers){
        //     layer->backward(grad);
        // }
    }

    template <typename T, int Width, int Height, int BatchSize>
    void train(datasets::Batch<T, Width, Height, BatchSize> batch){
        for (auto& dat : batch.data){
            auto flattened_data = utils::to_vector(dat.data);
            auto output = forward(dat);
            Eigen::VectorX<T> grad;
            for (int i = layers.size()-1; i >= 0; --i){
                layers[i]->backward(grad);
            }
        }
    }

