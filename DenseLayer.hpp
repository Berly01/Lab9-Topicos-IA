#pragma once
#include "functions.hpp"
#include "utils_cuda.hpp"
#include <iomanip>

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

    DenseLayer(size_t input_size, size_t output_size, const DenseLayerHyperparameters& params)
        : input_size_(input_size), output_size_(output_size), params_(params), gen_(std::random_device{}()) {

        // Inicializar pesos
        initialize_weights();

        // Inicializar bias
        if (params_.use_bias) {
            biases_ = Matrix<double>(output_size_, 1, 0.0);
            std::uniform_real_distribution<double> bias_dist(-0.1, 0.1);
            for (size_t i = 0; i < output_size_; ++i) {
                biases_[i][0] = bias_dist(gen_);
            }
        }

        // Inicializar gradientes
        grad_weights_ = Matrix<double>(output_size_, input_size_, 0.0);
        grad_biases_ = Matrix<double>(output_size_, 1, 0.0);
    }

    Matrix<double> forward(const Matrix<double>& input) {
        // Guardar entrada para backward pass
        cached_input_ = input;

        // Multiplicación matricial: input * weights
        Matrix<double> output = weights_ * input;

        // Agregar bias
        if (params_.use_bias) {
            output + biases_;
        }

        // Guardar salida antes de activación
        cached_linear_output_ = output;

        // Aplicar activación
        if (params_.use_activation) {
            if (params_.activation_func == Activation::RELU) {
                output = output.apply_function(relu);
            }
            else if (params_.activation_func == Activation::SIGMOID) {
                output = output.apply_function(sigmoid);
            }
            else if (params_.activation_func == Activation::TANH) {
                output = output.apply_function(tanhm);
            }
        }

        return output;
    }

    Matrix<double> backward(const Matrix<double>& grad_output) {
        Matrix<double> grad = grad_output;

        // Gradiente de activación
        if (params_.use_activation) {
            Matrix<double> activation_grad;
            if (params_.activation_func == Activation::RELU) {
                activation_grad = cached_linear_output_.apply_function(relu_deri);
            }
            else if (params_.activation_func == Activation::SIGMOID) {
                activation_grad = cached_linear_output_.apply_function(sigmoid_deri);
            }
            else if (params_.activation_func == Activation::TANH) {
                activation_grad = cached_linear_output_.apply_function(tanh_deri);
            }

            // Multiplicación elemento a elemento
            for (size_t i = 0; i < grad.rows(); ++i) {
                for (size_t j = 0; j < grad.cols(); ++j) {
                    grad[i][j] *= activation_grad[i][j];
                }
            }
        }

        // Gradiente de pesos: input^T * grad
        grad_weights_ =  grad * cached_input_.transpose();


        // Gradiente de bias
        if (params_.use_bias) {
            double bias_grad = 0.0;
            for (size_t i = 0; i < grad.rows(); ++i) {
                for (size_t j = 0; j < grad.cols(); ++j) {
                    bias_grad += grad[i][j];
                }
                grad_biases_[i][0] = bias_grad;
            }
        }

        // Gradiente de entrada: grad * weights^T
        Matrix<double> grad_input =  weights_.transpose() * grad;

        return grad_input;
    }

    void update_parameters() {
        // Actualizar pesos
        for (size_t i = 0; i < weights_.rows(); ++i) {
            for (size_t j = 0; j < weights_.cols(); ++j) {
                weights_[i][j] -= params_.learning_rate * grad_weights_[i][j];
            }
        }

        // Actualizar bias
        if (params_.use_bias) {
            for (size_t j = 0; j < biases_.cols(); ++j) {
                biases_[j][0] -= params_.learning_rate * grad_biases_[j][0];
            }
        }
    }

private:
    void initialize_weights() {
        if (params_.init == Initializer::N_XAVIER) {
            weights_ = xavier_normal_init(input_size_, output_size_, gen_);
        }
        else if (params_.init == Initializer::U_XAVIER) {
            weights_ = xavier_uniform_init(input_size_, output_size_, gen_);
        }
        else if (params_.init == Initializer::N_HE) {
            weights_ = he_normal_init(input_size_, output_size_, gen_);
        }
        else if (params_.init == Initializer::U_HE) {
            weights_ = he_uniform_init(input_size_, output_size_, gen_);
        }
        else {
            weights_ = random_init(input_size_, output_size_, -0.1, 0.1, gen_);
        }
    }

    size_t input_size_;
    size_t output_size_;
    DenseLayerHyperparameters params_;
    std::mt19937 gen_;

    Matrix<double> weights_;
    Matrix<double> biases_;
    Matrix<double> grad_weights_;
    Matrix<double> grad_biases_;

    // Cache para backward pass
    Matrix<double> cached_input_;
    Matrix<double> cached_linear_output_;
};
