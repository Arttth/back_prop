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
    virtual T computeDerivative(const T& val) = 0;
};

template <typename T>
class ReLuFunc : public ActivationFunc<T> {
public:
    T compute(const T& val) override {
        return val < static_cast<T>(0) ? static_cast<T>(0) : val;
    }

    T computeDerivative(const T& val) override {
       return val > 0 ? val : 0;
    }
};

template <typename T>
class InputFunc : public ActivationFunc<T> {
public:
    T compute(const T& val) override {
        return val;
    }

    T computeDerivative(const T& val) override {
        return 0;
    }
};

template <typename T>
class Neuron {
public:
    Neuron() = default;

    Neuron(size_t weightCount, const T& val, std::shared_ptr<ActivationFunc<T>> actFunc)
        : val_(val), actFunc_(std::move(actFunc))
    {
        weights_.reserve(weightCount);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < weightCount; ++i) {
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
        assert(input.size() == weights_.size());
        T linComb = bias_;
        for (size_t i = 0; i < input.size(); ++i) {
            linComb += input[i].getValue() * weights_[i];
        }
        z_ = linComb;
        val_ = actFunc_->compute(linComb);
    }

    size_t weightsCount() const { return weights_.size(); }

private:
    T val_{static_cast<T>(0)};
    T z_{0}; // значение лин. комб
    T bias_{0};
    T delta_{0};
    std::vector<T> weights_;
    std::shared_ptr<ActivationFunc<T>> actFunc_;
};

} // namespace nw
