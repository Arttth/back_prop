#pragma once

#include "Layer.h"
#include "LayerGen.h"
#include <vector>

namespace nw {

template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork(size_t layersCount) : 
        layersCount_(layersCount) {}
    
    void init(std::vector<InitLayer<T>> layers) {
        if (layers.size() <= 0)
            return;
       layers_.emplace_back(layers[0].neuronsCount_, 0, layers[0].actFunc_);
       for (size_t i = 0; i < layers.size(); ++i) {
            layers_.emplace_back(layers[i].neuronsCount_, layers[i-1].neuronsCount_, layers[i].actFunc_);
       }
    }

    void forward(const std::vector<T> input) {
        //static_assert(input.size() == layers_[0].size());
        
        layers_[0].setNeurons(input); 
        for (int i = 1; i < layers_.size(); ++i) {
            layers_[i].compute(layers_[i-1].neurons());
        }
    }

    std::vector<Neuron<T>> getOutput() {
        return layers_.back().neurons();
    }

    void printOutput() {
        for (size_t i = 0; i <  layers_.back().neurons().size(); ++i) {
            std::cout << "Output for " << i << " is " << layers_.back().neurons()[i].getValue() << std::endl;
        }
    }

    
    
private:
   size_t layersCount_; 
   std::vector<Layer<T>> layers_;
};

}

