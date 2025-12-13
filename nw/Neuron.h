#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cassert>

namespace nw {

template<typename T>
class ActivationFunc {
public:
    virtual ~ActivationFunc() = default; 
    virtual T compute(const T& val) = 0; 
};

template <typename T>
class ReLuFunc : public ActivationFunc<T> {
public:
    T compute(const T& val) override {
        return val < 0 ? 0 : val;
    }
}; 

template <typename T>
class InputFunc : public ActivationFunc<T> {
public:
    T compute(const T& val) override {
        return val;
    }
};


template <typename T>
class Neuron {
public:
    Neuron(size_t weightCount, const T& val, std::shared_ptr<ActivationFunc<T>> actFunc) :
        weightCount_(weightCount), val_(val), actFunc_(actFunc)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < weightCount_; ++i) {
            T randInitVal = static_cast<T>(dist(gen));
            weights_.emplace_back(randInitVal);        
        }
    }

    void setValue(const T& val) {
        val_ = val;
    }

    const T& getValue() const {
        return val_;
    } 


    void activate(const std::vector<Neuron<T>>& input) {
        assert(input.size() == weightCount_);
        T linComb = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            linComb += input[i].getValue()*weights_[i];
        }
        val_ = actFunc_->compute(linComb);
    }

    
private:
    size_t weightCount_;
    T val_;
    std::vector<T> weights_;
    std::shared_ptr<ActivationFunc<T>> actFunc_;
};


}
