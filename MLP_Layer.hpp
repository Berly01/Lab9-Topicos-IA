#ifndef MLP_Layer_HPP
#define MLP_Layer_HPP

#pragma once
#include <random>
#include "Matrix.hpp"
#include "functions.hpp"

class MLP_Layer {

public:
    Matrix<double> weights;
    Matrix<double> biases;
    Matrix<double> z;
    Matrix<double> activation;
    Matrix<double> cache_w;
    Matrix<double> cache_b;
    Matrix<double> m_w;
    Matrix<double> v_w;
    Matrix<double> m_b;
    Matrix<double> v_b;
    Matrix<double> mask;

    MLP_Layer() = default;

    explicit MLP_Layer(const size_t& input_size,
        const size_t& output_size,
        const Initializer& init,
        const double& bias_init,
        std::mt19937& gen)
        : weights(output_size, input_size, 0.0),
        biases(output_size, 1, bias_init),
        cache_w(output_size, input_size, 0.0),
        cache_b(output_size, 1, 0.0),
        m_w(output_size, input_size, 0.0),
        v_w(output_size, input_size, 0.0),
        m_b(output_size, 1, 0.0),
        v_b(output_size, 1, 0.0),
        mask(output_size, 1, 1.0) {
           
        if (init == Initializer::U_XAVIER)
            weights = xavier_uniform_init(input_size, output_size, gen);
        else if (init == Initializer::N_XAVIER)
            weights = xavier_normal_init(input_size, output_size, gen);
        else if (init == Initializer::U_HE)
            weights = he_uniform_init(input_size, output_size, gen);
        else if (init == Initializer::N_HE)
            weights = he_normal_init(input_size, output_size, gen);
        else
            weights = random_init(input_size, output_size, gen);
    }
};

#endif