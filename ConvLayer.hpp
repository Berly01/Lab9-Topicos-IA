#pragma once
#include <vector>
#include "functions.hpp"
#include "utils_cuda.hpp"

enum class PoolMode { MIN, MAX, AVG };

class ConvLayer {

public:
    struct ConvLayerHyperparameters {
        bool padding = false;
        bool pooling = false;
        bool activation = false;
        bool debug = false;
        bool measure = false;
        bool use_bias = true; 

        size_t padding_size = 1;
        size_t stride = 1;
        size_t pool_size = 2;
        double learning_rate = 0.001;
        PoolMode pool_mode = PoolMode::MAX;
        Activation activation_func = Activation::RELU;
        Initializer init = Initializer::RANDOM;
    };

    struct ForwardCache {
        std::vector<Matrix<double>> input_channels;
        std::vector<Matrix<double>> padded_inputs;
        std::vector<Matrix<double>> conv_outputs;
        std::vector<Matrix<double>> activated_outputs;
        std::vector<Matrix<double>> pooled_outputs;
        std::vector<std::vector<std::pair<size_t, size_t>>> pool_indices; // Para max pooling
    };

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    std::mt19937 gen;

    std::vector<std::vector<Matrix<double>>> filters_;
    std::vector<double> biases_;

    std::vector<std::vector<Matrix<double>>> grad_filters_;
    std::vector<double> grad_biases_;

    ForwardCache cache_;

    ConvLayerHyperparameters h_;

public:
    ConvLayer(
        const size_t& _in_channels,
        const size_t& _out_channels,
        const size_t& _kernel_size,
        const ConvLayerHyperparameters& _h);

    std::vector<Matrix<double>> forward(
        const std::vector<Matrix<double>>& _input_channels);

    std::vector<Matrix<double>> backward(
        const std::vector<Matrix<double>>& _grad_output);

    void update_parameters();

    std::vector<std::vector<Matrix<double>>> get_filters() const { return filters_; }

    size_t get_stride_size() const { return h_.stride; }

    size_t get_kernel_size() const { return kernel_size_; }

    size_t get_padding_size() const { return h_.padding ? h_.padding_size : 0; }

    size_t get_pool_size() const { return h_.pooling ? h_.pool_size : 0; }

    bool uses_pooling() const { return h_.pooling; }

private:
    std::vector<Matrix<double>> create_kernel(
        const size_t& _rows,
        const size_t& _cols,
        const size_t& _in_channels,
        const size_t& _out_channels,
        const Initializer& _init,
        std::mt19937& gen) const;

    Matrix<double> padding(
        const Matrix<double>& _m,
        const size_t& _padding,
        const double& _value = 0.0) const;

    Matrix<double> activacion(
        const Matrix<double>& _m,
        const Activation& _f) const;

    Matrix<double> pooling(
        const Matrix<double>& _m,
        const size_t& _pool_size,
        const PoolMode& _pool_mode,
        std::vector<std::pair<size_t, size_t>>& pool_indices) const;

    Matrix<double> convolution(
        const std::vector<Matrix<double>>& _input_channels,
        const std::vector<Matrix<double>>& _kernels,
        const size_t& _padding,
        const size_t& _stride) const;

    Matrix<double> activation_derivative(
        const Matrix<double>& _m,
        const Activation& _f) const;

    Matrix<double> pooling_backward(
        const Matrix<double>& _grad_output,
        const Matrix<double>& _original_input,
        const std::vector<std::pair<size_t, size_t>>& _pool_indices,
        const size_t& _pool_size,
        const PoolMode& _pool_mode) const;

    std::vector<Matrix<double>> convolution_backward_input(
        const Matrix<double>& _grad_output,
        const std::vector<Matrix<double>>& _kernels,
        const size_t& _input_rows,
        const size_t& _input_cols,
        const size_t& _padding,
        const size_t& _stride) const;

    std::vector<Matrix<double>> convolution_backward_kernel(
        const Matrix<double>& _grad_output,
        const std::vector<Matrix<double>>& _input_channels,
        const size_t& _stride) const;

    Matrix<double> remove_padding(
        const Matrix<double>& _m,
        const size_t& _padding_size) const;
};


ConvLayer::ConvLayer(
    const size_t& _in_channels,
    const size_t& _out_channels,
    const size_t& _kernel_size,
    const ConvLayerHyperparameters& _h)
    : gen(std::random_device{}()),
    in_channels_(_in_channels),
    out_channels_(_out_channels),
    kernel_size_(_kernel_size),
    h_(_h) {

    for (size_t f = 0; f < out_channels_; ++f)
        filters_.emplace_back(create_kernel(kernel_size_, kernel_size_, in_channels_, out_channels_, h_.init, gen));

    if (h_.use_bias) {
        biases_.resize(out_channels_, 0.0);
        std::uniform_real_distribution<double> bias_init(-0.01, 0.01);
        for (size_t i = 0; i < out_channels_; ++i) {
            biases_[i] = bias_init(gen);
        }
    }

    grad_filters_.resize(out_channels_);
    for (size_t f = 0; f < out_channels_; ++f) {
        grad_filters_[f].resize(in_channels_);
        for (size_t c = 0; c < in_channels_; ++c) {
            grad_filters_[f][c] = Matrix<double>(kernel_size_, kernel_size_, 0.0);
        }
    }
    grad_biases_.resize(out_channels_, 0.0);

    if (h_.debug) {
        std::cout << "KERNELS Y BIASES INICIALIZADOS\n";
        int k = 1;
        for (size_t f = 0; f < filters_.size(); ++f) {
            std::cout << "K " << k << " (bias: " << (h_.use_bias ? biases_[f] : 0.0) << ")\n";
            int c = 1;
            for (const auto& channel : filters_[f]) {
                std::cout << "CANAL " << c << '\n';
                std::cout << '\n';
                ++c;
            }
            std::cout << '\n';
            ++k;
        }
    }
}

std::vector<Matrix<double>> ConvLayer::create_kernel(
    const size_t& _rows,
    const size_t& _cols,
    const size_t& _in_channels,
    const size_t& _out_channels,
    const Initializer& _init,
    std::mt19937& _gen) const {

    std::vector<Matrix<double>> kernel;
    const auto fa_in = _in_channels * _rows * _cols;
    const auto fa_out = _out_channels * _rows * _cols;

    if (_init == Initializer::U_HE) {
        for (size_t c = 0; c < _in_channels; ++c)
            kernel.emplace_back(he_uniform_init(_rows, _cols, fa_in, _gen));
    }
    else if (_init == Initializer::N_HE) {
        for (size_t c = 0; c < _in_channels; ++c)
            kernel.emplace_back(he_normal_init(_rows, _cols, fa_in, _gen));
    }
    else if (_init == Initializer::U_XAVIER) {
        for (size_t c = 0; c < _in_channels; ++c)
            kernel.emplace_back(xavier_uniform_init(_rows, _cols, fa_in, fa_out, _gen));
    }
    else if (_init == Initializer::N_XAVIER) {
        for (size_t c = 0; c < _in_channels; ++c)
            kernel.emplace_back(xavier_normal_init(_rows, _cols, fa_in, fa_out, _gen));
    }
    else {
        for (size_t c = 0; c < _in_channels; ++c)
            kernel.emplace_back(random_init(_rows, _cols, -0.1, 0.1, _gen));
    }

    return kernel;
}

Matrix<double> ConvLayer::padding(
    const Matrix<double>& _m,
    const size_t& _padding_size,
    const double& _value) const {

    const size_t NEW_ROWS = _m.rows() + 2 * _padding_size;
    const size_t NEW_COLS = _m.cols() + 2 * _padding_size;
    Matrix<double> padded(NEW_ROWS, NEW_COLS, _value);

    for (size_t i = 0; i < _m.rows(); ++i)
        for (size_t j = 0; j < _m.cols(); ++j)
            padded[i + _padding_size][j + _padding_size] = _m[i][j];

    return padded;
}

Matrix<double> ConvLayer::remove_padding(
    const Matrix<double>& _m,
    const size_t& _padding_size) const {

    const size_t NEW_ROWS = _m.rows() - 2 * _padding_size;
    const size_t NEW_COLS = _m.cols() - 2 * _padding_size;
    Matrix<double> unpadded(NEW_ROWS, NEW_COLS);

    for (size_t i = 0; i < NEW_ROWS; ++i)
        for (size_t j = 0; j < NEW_COLS; ++j)
            unpadded[i][j] = _m[i + _padding_size][j + _padding_size];

    return unpadded;
}

Matrix<double> ConvLayer::activacion(
    const Matrix<double>& _m,
    const Activation& _f) const {

    if (_f == Activation::RELU)
        return CU::apply_function(_m, ReluFunctor());
    else if (_f == Activation::SIGMOID)
        return CU::apply_function(_m, SigmoidFunctor());
    else
        return CU::apply_function(_m, TanhFunctor());
}

Matrix<double> ConvLayer::activation_derivative(
    const Matrix<double>& _m,
    const Activation& _f) const {

    if (_f == Activation::RELU)
        return CU::apply_function(_m, DReluFunctor());
    else if (_f == Activation::SIGMOID)
        return CU::apply_function(_m, DSigmoidFunctor());
    else
        return CU::apply_function(_m, DTanhFunctor());
}

Matrix<double> ConvLayer::pooling(
    const Matrix<double>& _m,
    const size_t& _pool_size,
    const PoolMode& _pool_mode,
    std::vector<std::pair<size_t, size_t>>& pool_indices) const {

    const size_t OUT_ROWS = _m.rows() / _pool_size;
    const size_t OUT_COLS = _m.cols() / _pool_size;
    Matrix<double> output(OUT_ROWS, OUT_COLS, 0);

    pool_indices.clear();
    pool_indices.resize(OUT_ROWS * OUT_COLS);

    for (size_t i = 0; i < OUT_ROWS; ++i) {
        for (size_t j = 0; j < OUT_COLS; ++j) {
            double val{};
            size_t max_i = 0;
            size_t max_j = 0;

            if (_pool_mode == PoolMode::MAX) val = std::numeric_limits<double>::lowest();
            else if (_pool_mode == PoolMode::MIN) val = std::numeric_limits<double>::max();
            else val = 0.0;

            if (_pool_mode == PoolMode::MAX) {
                for (size_t pi = 0; pi < _pool_size; ++pi) {
                    for (size_t pj = 0; pj < _pool_size; ++pj) {
                        const double current = _m[i * _pool_size + pi][j * _pool_size + pj];
                        if (current > val) {
                            val = current;
                            max_i = i * _pool_size + pi;
                            max_j = j * _pool_size + pj;
                        }
                    }
                }
                pool_indices[i * OUT_COLS + j] = { max_i, max_j };
            }
            else if (_pool_mode == PoolMode::MIN) {
                for (size_t pi = 0; pi < _pool_size; ++pi) {
                    for (size_t pj = 0; pj < _pool_size; ++pj) {
                        const double current = _m[i * _pool_size + pi][j * _pool_size + pj];
                        if (current < val) {
                            val = current;
                            max_i = i * _pool_size + pi;
                            max_j = j * _pool_size + pj;
                        }
                    }
                }
                pool_indices[i * OUT_COLS + j] = { max_i, max_j };
            }
            else {
                for (size_t pi = 0; pi < _pool_size; ++pi) {
                    for (size_t pj = 0; pj < _pool_size; ++pj) {
                        const double current = _m[i * _pool_size + pi][j * _pool_size + pj];
                        val += current;
                    }
                }
                val /= static_cast<double>(_pool_size * _pool_size);
            }

            output[i][j] = val;
        }
    }

    return output;
}

Matrix<double> ConvLayer::pooling_backward(
    const Matrix<double>& _grad_output,
    const Matrix<double>& _original_input,
    const std::vector<std::pair<size_t, size_t>>& _pool_indices,
    const size_t& _pool_size,
    const PoolMode& _pool_mode) const {

    Matrix<double> grad_input(_original_input.rows(), _original_input.cols(), 0.0);

    for (size_t i = 0; i < _grad_output.rows(); ++i) {
        for (size_t j = 0; j < _grad_output.cols(); ++j) {
            const double grad_val = _grad_output[i][j];

            if (_pool_mode == PoolMode::MAX || _pool_mode == PoolMode::MIN) {
                // Solo propagar el gradiente al elemento que fue seleccionado
                const auto& indices = _pool_indices[i * _grad_output.cols() + j];
                grad_input[indices.first][indices.second] += grad_val;
            }
            else { // AVG pooling
                // Distribuir el gradiente uniformemente
                const double distributed_grad = grad_val / (_pool_size * _pool_size);
                for (size_t pi = 0; pi < _pool_size; ++pi) {
                    for (size_t pj = 0; pj < _pool_size; ++pj) {
                        grad_input[i * _pool_size + pi][j * _pool_size + pj] += distributed_grad;
                    }
                }
            }
        }
    }

    return grad_input;
}

Matrix<double> ConvLayer::convolution(
    const std::vector<Matrix<double>>& _input_channels,
    const std::vector<Matrix<double>>& _kernels,
    const size_t& _padding,
    const size_t& _stride) const {


    if (_input_channels.size() != _kernels.size())
        throw std::invalid_argument("Cantidad de canales no coincide con kernels del filtro");

    Matrix<double> result;
    bool first_conv = true;

    auto i_channel = _input_channels.begin();
    auto i_kernel = _kernels.begin();

    for (; i_channel != _input_channels.end(); ++i_channel, ++i_kernel) {
        auto padded = h_.padding ? padding(*i_channel, _padding) : *i_channel;
        const auto K = (*i_kernel).rows();
        const size_t OUT_ROWS = (padded.rows() - K) / _stride + 1;
        const size_t OUT_COLS = (padded.cols() - K) / _stride + 1;

        Matrix<double> conv(OUT_ROWS, OUT_COLS, 0.0);

        for (size_t i = 0; i < OUT_ROWS; ++i) {
            for (size_t j = 0; j < OUT_COLS; ++j) {
                double acc = 0.0;
                for (size_t ki = 0; ki < K; ++ki)
                    for (size_t kj = 0; kj < K; ++kj)
                        acc += (*i_kernel)[ki][kj] * padded[i * _stride + ki][j * _stride + kj];
                conv[i][j] = acc;
            }
        }

        result = first_conv ? conv : result + conv;
        if (first_conv) first_conv = false;
    }

    return result;
}

std::vector<Matrix<double>> ConvLayer::convolution_backward_input(
    const Matrix<double>& _grad_output,
    const std::vector<Matrix<double>>& _kernels,
    const size_t& _input_rows,
    const size_t& _input_cols,
    const size_t& _padding,
    const size_t& _stride) const {


    std::vector<Matrix<double>> grad_input;

    for (size_t c = 0; c < in_channels_; ++c) {
        grad_input.emplace_back(_input_rows, _input_cols, 0.0);
    }

    for (size_t c = 0; c < in_channels_; ++c) {
        const auto& kernel = _kernels[c];

        Matrix<double> padded_grad_input = h_.padding ?
            padding(grad_input[c], _padding) : grad_input[c];

        for (size_t i = 0; i < _grad_output.rows(); ++i) {
            for (size_t j = 0; j < _grad_output.cols(); ++j) {
                const double grad_val = _grad_output[i][j];

                for (size_t ki = 0; ki < kernel.rows(); ++ki) {
                    for (size_t kj = 0; kj < kernel.cols(); ++kj) {
                        const size_t input_i = i * _stride + ki;
                        const size_t input_j = j * _stride + kj;

                        if (input_i < padded_grad_input.rows() && input_j < padded_grad_input.cols()) {
                            const double kernel_val = kernel[kernel.rows() - 1 - ki][kernel.cols() - 1 - kj];
                            padded_grad_input[input_i][input_j] += grad_val * kernel_val;
                        }
                    }
                }
            }
        }

        grad_input[c] = h_.padding ?
            remove_padding(padded_grad_input, _padding) : padded_grad_input;
    }

    return grad_input;
}

std::vector<Matrix<double>> ConvLayer::convolution_backward_kernel(
    const Matrix<double>& _grad_output,
    const std::vector<Matrix<double>>& _input_channels,
    const size_t& _stride) const {

    
    std::vector<Matrix<double>> grad_kernels;
   
    for (size_t c = 0; c < in_channels_; ++c) {
        grad_kernels.emplace_back(kernel_size_, kernel_size_, 0.0);
    }

    for (size_t c = 0; c < in_channels_; ++c) {
        const auto& input_channel = _input_channels[c];
        auto padded_input = h_.padding ?
            padding(input_channel, h_.padding_size) : input_channel;

        for (size_t ki = 0; ki < kernel_size_; ++ki) {
            for (size_t kj = 0; kj < kernel_size_; ++kj) {
                double grad_sum = 0.0;

                for (size_t i = 0; i < _grad_output.rows(); ++i) {
                    for (size_t j = 0; j < _grad_output.cols(); ++j) {
                        const size_t input_i = i * _stride + ki;
                        const size_t input_j = j * _stride + kj;

                        if (input_i < padded_input.rows() && input_j < padded_input.cols()) {
                            grad_sum += _grad_output[i][j] * padded_input[input_i][input_j];
                        }
                    }
                }

                grad_kernels[c][ki][kj] = grad_sum;
            }
        }
    }

    return grad_kernels;
}

std::vector<Matrix<double>> ConvLayer::forward(
    const std::vector<Matrix<double>>& _input_channels) {

    cache_.input_channels = _input_channels;
    cache_.padded_inputs.clear();
    cache_.conv_outputs.clear();
    cache_.activated_outputs.clear();
    cache_.pooled_outputs.clear();
    cache_.pool_indices.clear();

    std::vector<Matrix<double>> outputs;

    for (const auto& input : _input_channels) {
        cache_.padded_inputs.emplace_back(
            h_.padding ? padding(input, h_.padding_size) : input
        );
    }

    int k = 1;
    for (const auto& kernels : filters_) {
        if (h_.debug) std::cout << "*********KERNEL " << k << "*********\n";

        auto features = convolution(cache_.padded_inputs, kernels, h_.padding_size, h_.stride);

        if (h_.use_bias) {
            const double bias_val = biases_[k - 1];
            for (size_t i = 0; i < features.rows(); ++i) {
                for (size_t j = 0; j < features.cols(); ++j) {
                    features[i][j] += bias_val;
                }
            }
        }

        cache_.conv_outputs.emplace_back(features);

        if (h_.activation) {
            features = activacion(features, h_.activation_func);
            cache_.activated_outputs.emplace_back(features);
        }
        else {
            cache_.activated_outputs.emplace_back(features);
        }

        if (h_.pooling) {
            std::vector<std::pair<size_t, size_t>> pool_idx;
            features = pooling(features, h_.pool_size, h_.pool_mode, pool_idx);
            cache_.pool_indices.emplace_back(pool_idx);
            cache_.pooled_outputs.emplace_back(features);
        }
        else {
            cache_.pooled_outputs.emplace_back(features);
        }

        outputs.emplace_back(features);
        ++k;
    }

    if (h_.debug) {
        std::cout << "\nRESULTADO FINAL: " << outputs[0].rows() << 'x' << outputs[0].cols() << 'x' << outputs.size() << '\n';
        int c = 1;
        for (const auto& m : outputs) {
            std::cout << "CANAL " << c << '\n';
            //m.print(12);
            std::cout << '\n';
            ++c;
        }
    }

    return outputs;
}

std::vector<Matrix<double>> ConvLayer::backward(
    const std::vector<Matrix<double>>& _grad_output) {

    for (auto& grad_filter : grad_filters_) {
        for (auto& grad_channel : grad_filter) {
            grad_channel = Matrix<double>(kernel_size_, kernel_size_, 0.0);
        }
    }

    std::fill(grad_biases_.begin(), grad_biases_.end(), 0.0);

    std::vector<Matrix<double>> grad_input;

    for (size_t f = 0; f < out_channels_; ++f) {
        Matrix<double> current_grad = _grad_output[f];

        if (h_.pooling) {
            current_grad = pooling_backward(
                current_grad,
                cache_.activated_outputs[f],
                cache_.pool_indices[f],
                h_.pool_size,
                h_.pool_mode
            );
        }

        if (h_.activation) {
            const auto activation_grad = activation_derivative(
                cache_.conv_outputs[f],
                h_.activation_func
            );

            for (size_t i = 0; i < current_grad.rows(); ++i) {
                for (size_t j = 0; j < current_grad.cols(); ++j) {
                    current_grad[i][j] *= activation_grad[i][j];
                }
            }
        }

        if (h_.use_bias) {
            double bias_grad = 0.0;
            for (size_t i = 0; i < current_grad.rows(); ++i) {
                for (size_t j = 0; j < current_grad.cols(); ++j) {
                    bias_grad += current_grad[i][j];
                }
            }
            grad_biases_[f] = bias_grad;
        }

        const auto kernel_grads = convolution_backward_kernel(
            current_grad,
            cache_.input_channels,
            h_.stride
        );

        for (size_t c = 0; c < in_channels_; ++c) {
            grad_filters_[f][c] = grad_filters_[f][c] + kernel_grads[c];
        }

        const auto input_grads = convolution_backward_input(
            current_grad,
            filters_[f],
            cache_.input_channels[0].rows(),
            cache_.input_channels[0].cols(),
            h_.padding_size,
            h_.stride
        );

        if (grad_input.empty()) {
            grad_input = input_grads;
        }
        else {
            for (size_t c = 0; c < in_channels_; ++c) {
                grad_input[c] = grad_input[c] + input_grads[c];
            }
        }
    }

    if (h_.debug) {
        std::cout << "\n=== GRADIENTES CALCULADOS ===\n";
        for (size_t f = 0; f < out_channels_; ++f) {
            std::cout << "Filter " << f << " bias gradient: " << grad_biases_[f] << '\n';
            for (size_t c = 0; c < in_channels_; ++c) {
                std::cout << "Filter " << f << " Channel " << c << " kernel gradient:\n";
                //grad_filters_[f][c].print(12);
                std::cout << '\n';
            }
        }
    }

    return grad_input;
}

void ConvLayer::update_parameters() {
    for (size_t f = 0; f < out_channels_; ++f) {
        for (size_t c = 0; c < in_channels_; ++c) {
            for (size_t i = 0; i < kernel_size_; ++i) {
                for (size_t j = 0; j < kernel_size_; ++j) {
                    filters_[f][c][i][j] -= h_.learning_rate * grad_filters_[f][c][i][j];
                }
            }
        }
    }

    if (h_.use_bias) {
        for (size_t f = 0; f < out_channels_; ++f) {
            biases_[f] -= h_.learning_rate * grad_biases_[f];
        }
    }

    if (h_.debug) {
        std::cout << "\n=== PARAMETROS ACTUALIZADOS ===\n";
        for (size_t f = 0; f < out_channels_; ++f) {
            std::cout << "Filter " << f << " bias: " << biases_[f] << '\n';
        }
    }
}
