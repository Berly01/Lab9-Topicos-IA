#pragma once
#include "functions.hpp"
#include "utils_cuda.hpp"

class DenseLayer {
public:

    struct DenseLayerHyperparameters {
        bool use_bias = true;
        bool use_activation = true;
        bool debug = false;
        double learning_rate = 0.001;
        Activation activation_func = Activation::RELU;
        Initializer init = Initializer::N_XAVIER;
    };

    DenseLayer(size_t input_size, size_t output_size, const DenseLayerHyperparameters& params);

    Matrix<double> forward(const Matrix<double>& input);

    Matrix<double> backward(const Matrix<double>& grad_output);

    void update_parameters();

private:
    void initialize_weights();

    void initialize_biases();

    size_t input_size_;
    size_t output_size_;
    DenseLayerHyperparameters params_;
    std::mt19937 gen_;

    Matrix<double> weights_;
    Matrix<double> biases_;
    Matrix<double> grad_weights_;
    Matrix<double> grad_biases_;

    Matrix<double> cached_input_;
    Matrix<double> cached_linear_output_;
};

DenseLayer::DenseLayer(size_t input_size, size_t output_size, const DenseLayerHyperparameters& params)
    : input_size_(input_size), output_size_(output_size), params_(params), gen_(std::random_device{}()) {

    initialize_weights();

    if (params_.use_bias) initialize_biases();

    grad_weights_ = Matrix<double>(output_size_, input_size_, 0.0);
    grad_biases_ = Matrix<double>(output_size_, 1, 0.0);
}

Matrix<double> DenseLayer::forward(const Matrix<double>& input) {

    cached_input_ = input;

    Matrix<double> output = CU::dot_product(weights_, input);

    if (params_.use_bias) output = CU::basic_oper(output, biases_, Operation::ADD);

    cached_linear_output_ = output;

    if (params_.use_activation) {
        if (params_.activation_func == Activation::RELU) {
            output = CU::apply_function(output, ReluFunctor());
        }
        else if (params_.activation_func == Activation::SIGMOID) {
            output = CU::apply_function(output, SigmoidFunctor());
        }
        else if (params_.activation_func == Activation::TANH) {
            output = CU::apply_function(output, TanhFunctor());
        }
    }

    return output;
}

Matrix<double> DenseLayer::backward(const Matrix<double>& grad_output) {

    Matrix<double> grad = grad_output;

    if (params_.use_activation) {
        Matrix<double> activation_grad;
        if (params_.activation_func == Activation::RELU) {
            activation_grad = CU::apply_function(cached_linear_output_, DReluFunctor());
        }
        else if (params_.activation_func == Activation::SIGMOID) {
            activation_grad = CU::apply_function(cached_linear_output_, DSigmoidFunctor());
        }
        else if (params_.activation_func == Activation::TANH) {
            activation_grad = CU::apply_function(cached_linear_output_, DTanhFunctor());
        }

        for (size_t i = 0; i < grad.rows(); ++i) {
            for (size_t j = 0; j < grad.cols(); ++j) {
                grad[i][j] *= activation_grad[i][j];
            }
        }
    }

    grad_weights_ = CU::dot_product(grad, cached_input_.transpose());

   
    if (params_.use_bias) {
        for (size_t j = 0; j < grad.cols(); ++j) {
            double bias_grad = 0.0;
            for (size_t i = 0; i < grad.rows(); ++i) {
                bias_grad += grad[i][j];
            }
            grad_biases_[j][0] = bias_grad;
        }
    }

    Matrix<double> grad_input = CU::dot_product(weights_.transpose(), grad);

    return grad_input;
}

void DenseLayer::update_parameters() {

    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            weights_[i][j] -= params_.learning_rate * grad_weights_[i][j];
        }
    }

    if (params_.use_bias) {
        for (size_t j = 0; j < biases_.cols(); ++j) {
            biases_[0][j] -= params_.learning_rate * grad_biases_[0][j];
        }
    }
}

void DenseLayer::initialize_weights() {
    if (params_.init == Initializer::N_XAVIER) {
        weights_ = xavier_uniform_init(input_size_, output_size_, gen_);
    }
    else if (params_.init == Initializer::U_XAVIER) {
        weights_ = xavier_normal_init(input_size_, output_size_, gen_);
    }
    else if (params_.init == Initializer::N_HE) {
        weights_ = he_uniform_init(input_size_, output_size_, gen_);
    }
    else if (params_.init == Initializer::U_HE) {
        weights_ = he_normal_init(input_size_, output_size_, gen_);
    }
    else {
        weights_ = random_init(input_size_, output_size_, gen_);
    }
}

void DenseLayer::initialize_biases() {
    biases_ = Matrix<double>(output_size_, 1, 0.0);
    std::uniform_real_distribution<> bias_dist(-0.01, 0.01);

    for (size_t i = 0; i < output_size_; ++i)
        biases_[i][0] = bias_dist(gen_);

}
