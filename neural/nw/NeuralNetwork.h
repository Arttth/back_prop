#pragma once

#include "Layer.h"
#include "LayerGen.h"
#include <vector>
#include <iostream>
#include <cassert>

namespace nw {

template <typename T>
class NeuralNetwork {
public:
    explicit NeuralNetwork(size_t /*layersCountHint*/ = 0) {}

    void init(const std::vector<InitLayer<T>>& layout) {
        if (layout.empty()) throw std::runtime_error("NeuralNetwork::init: empty layout");
        layout_ = layout;
    }

    void forward(const std::vector<T>& input) {
        if (layout_.empty()) throw std::runtime_error("NeuralNetwork::forward: network is not initialized");

        layers_.clear();
        layers_.emplace_back(input); 

        for (size_t i = 1; i < layout_.size(); ++i) {
            size_t prevIndex = layers_.size() - 1;
            const auto& spec = layout_[i];
            layers_.emplace_back(spec.neuronsCount_, layers_[prevIndex].size(), spec.actFunc_);
            layers_.back().compute(layers_[prevIndex].neurons());
        }
    }

    // TODO: class, mse until
    T errorFunc(const std::vector<T>& label, const std::vector<T>& output) {
        assert(label.size() == output().size());
        T error = 0;
        for (size_t i = 0; i < label.size(); ++i) {
            error += (label[i]-output[i])*(label[i]-output[i]);
        }
        return error/label.size()+1;
    }

    void backward(const std::vector<T>& label) {
         T err = errorFunc(label, layers_.back().neuronsVec()); 

    }

    void printOutput() const {
        if (layers_.empty()) {
            std::cout << "<empty network>" << std::endl;
            return;
        }
        const auto& out = layers_.back().neurons();
        std::cout << "Output (" << out.size() << "): ";
        for (const auto& n : out) {
            std::cout << n.getValue() << ' ';
        }
        std::cout << std::endl;
    }

private:
    std::vector<InitLayer<T>> layout_;
    std::vector<Layer<T>> layers_;
};

} // namespace nw
