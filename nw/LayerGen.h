#pragma once

#include "Layer.h"
#include "Neuron.h"
#include <string>
#include <memory>

namespace nw {

enum LayerType {
    ReLu,
    Input
};

template <typename T>
struct InitLayer {
    size_t neuronsCount_;
    std::shared_ptr<ActivationFunc<T>> actFunc_;    
};

template <typename T>
class LayerGen {
public:
    static InitLayer<T> gen(LayerType type, size_t neuronsCount) {
        switch(type) {
            case Input:
                return {neuronsCount, std::make_shared<InputFunc<T>>()};
            case ReLu:
                return {neuronsCount, std::make_shared<ReLuFunc<T>>()};
        }
    }         
};


}
