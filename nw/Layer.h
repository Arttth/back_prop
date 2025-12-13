#pragma once

#include "Neuron.h"
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>
#include <memory>

namespace nw {


template <typename T>
class Layer {
public:
    Layer(size_t neuronsCount, size_t inputNeuronsCount, std::shared_ptr<ActivationFunc<T>> actFunc) :
        neuronsCount_(neuronsCount), inputNeuronsCount_(inputNeuronsCount), actFunc_(actFunc) 
    {
        for (size_t i = 0; i < neurons_.size(); ++i) {
            neurons_.emplace_back(inputNeuronsCount_, 0, actFunc);
        }  
    }

    Layer(const std::vector<T>& input) {
        for (size_t i = 0; i < input.size(); ++i) {
            neurons_.emplace_back(0, input[i],std::make_shared<InputFunc<T>>());
        }
        neuronsCount_ = input.size();
        inputNeuronsCount_ = 0;
    }

    void compute(const std::vector<Neuron<T>> neurons) {
       std::cout << "Compute neurons for Layer" << std::endl;
       for (size_t i = 0; i < neurons_.size(); ++i) {
            neurons_[i].activate(neurons);
       } 
    }


    void setNeurons(const std::vector<T>& neurons) {
        if (neurons.size() != neuronsCount_)
            throw std::runtime_error("Layer: failed to setNeurons");
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons_.emplace_back(0,neurons[i], std::make_shared<InputFunc<T>>());
        }
    }

    const std::vector<Neuron<T>>& neurons() const {
        return neurons_;
    }

    size_t size() const {
        return neuronsCount_;
    }
private:
    std::vector<Neuron<T>> neurons_;
    size_t neuronsCount_;
    size_t inputNeuronsCount_;
    std::shared_ptr<ActivationFunc<T>> actFunc_;

};

}
