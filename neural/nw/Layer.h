#pragma once

#include "Neuron.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <cassert>

namespace nw {

template <typename T>
class Layer {
public:
    Layer(size_t neuronsCount, size_t inputNeuronsCount, std::shared_ptr<ActivationFunc<T>> actFunc)
        : inputNeuronsCount_(inputNeuronsCount), actFunc_(std::move(actFunc))
    {
        neurons_.reserve(neuronsCount);
        for (size_t i = 0; i < neuronsCount; ++i) {
            neurons_.emplace_back(inputNeuronsCount_, T{}, actFunc_);
        }
    }

    explicit Layer(const std::vector<T>& input) {
        neurons_.reserve(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            neurons_.emplace_back(0, input[i], std::make_shared<InputFunc<T>>());
        }
        inputNeuronsCount_ = 0;
        actFunc_ = std::make_shared<InputFunc<T>>();
    }

    void compute(const std::vector<Neuron<T>>& prev){
        if (neurons_.empty()) return;

        if (neurons_[0].weightsCount() != 0 && neurons_[0].weightsCount() != prev.size()) {
            std::cout << neurons_[0].weightsCount() << " " << prev.size() << std::endl;
            throw std::runtime_error("Layer::compute: input size mismatch");
        }

        for (auto& n : neurons_) {
            n.activate(prev);
        }
    }

    void computeOutputDelta(const std::vector<T>& expected) {
        
        for (int i = 0; i < neurons_.size(); ++i) {
            
        } 
    }

    const std::vector<Neuron<T>>& neurons() const { return neurons_; }

    const std::vector<T>& neuronsVec() const {
        std::vector<T> vec;
        for (auto& neuron : neurons_) {
            vec.emplace_back(neuron.getValue());
        }

        return vec;
    };

    size_t size() const { return neurons_.size(); }

private:
    std::vector<Neuron<T>> neurons_;
    size_t inputNeuronsCount_{0};
    std::shared_ptr<ActivationFunc<T>> actFunc_;
};

} // namespace nw
