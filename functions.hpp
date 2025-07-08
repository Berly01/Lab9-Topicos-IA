#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.hpp"
#include <random>

struct TrainingMetrics {
    std::vector<double> train_losses;
    std::vector<double> train_accuracies;

    void print_epoch_metrics(size_t epoch, double train_loss, double train_accuracy) const {
        std::cout << "Epoca " << epoch
            << " | Perdida: " << train_loss
            << " | Presicion: " << train_acc * 100 << "%"
            << std::endl;
    }
};

enum class Activation { SIGMOID, RELU, TANH };

enum class Initializer { U_XAVIER, N_XAVIER, U_HE, N_HE, RANDOM };

enum class Optimizer { ADAM, RMS_PROP, NONE };


struct SigmoidFunctor {
    __device__ double operator()(const double& x) const {
        return 1.0 / (1.0 + exp(-x));
    }
    SigmoidFunctor() = default;
};
struct DSigmoidFunctor {
    __device__ double operator()(const double& x) const {
        auto f = SigmoidFunctor()(x);
        return f * (1.0 - f);
    }
    DSigmoidFunctor() = default;
};
struct ReluFunctor {
    __device__ double operator()(const double& x) const {
        return x > 0.0 ? x : 0.0;
    }
    ReluFunctor() = default;
};
struct DReluFunctor {
    __device__ double operator()(const double& x) const {
        return x > 0.0 ? 1.0 : 0.0;
    }
    DReluFunctor() = default;
};
struct TanhFunctor {
    __device__ double operator()(const double& x) const {
        return (1.0 - std::exp(-2 * x)) / (1.0 + std::exp(-2 * x));
    }
    TanhFunctor() = default;
};
struct DTanhFunctor {
    __device__ double operator()(const double& x) const {
        auto f = TanhFunctor()(x);
        return 1.0 - f * f;
    }
    DTanhFunctor() = default;
};

struct SquareFunctor {
    __device__ double operator()(const double& x) const {
        return x * x;
    }
    SquareFunctor() = default;
};

struct SoftmaxFunctor {
    Matrix<double> operator()(const Matrix<double>& m) const {
        const auto ROWS = m.rows();

        Matrix<double> result(ROWS, 1);

        double max_val = m[0][0];
        for (size_t i = 1; i < ROWS; ++i)
            if (m[i][0] > max_val) max_val = m[i][0];

        double sum_exp = 0.0;
        for (size_t i = 0; i < ROWS; ++i) {
            result[i][0] = std::exp(m[i][0] - max_val);
            sum_exp += result[i][0];
        }

        for (size_t i = 0; i < ROWS; ++i)
            result[i][0] /= sum_exp;

        return result;
    }
    SoftmaxFunctor() = default;
};


inline double sigmoid(const double& _x) {
    return 1.0 / (1.0 + exp(-_x));
}

inline double sigmoid_deri(const double& _x) {
    const auto f = sigmoid(_x);
    return f * (1.0 - f);
}

inline double relu(const double& _x) {
    return std::max(0.0, _x);
}

inline double relu_deri(const double& _x) {
    return _x > 0.0 ? 1.0 : 0.0;
}


inline double tanhm(const double& _x) {
    return (1.0 - std::exp(-2 * _x)) / (1.0 + std::exp(-2 * _x));
}


inline double tanh_deri(const double& _x) {
    const auto f = tanhm(_x);
    return 1.0 - f * f;
}

inline Matrix<double> softmax(const Matrix<double>& input) {
    Matrix<double> result(input.rows(), input.cols());

    for (size_t i = 0; i < input.rows(); ++i) {
        double max_val = input[i][0];
        for (size_t j = 1; j < input.cols(); ++j) {
            if (input[i][j] > max_val) {
                max_val = input[i][j];
            }
        }

        double sum = 0.0;
        for (size_t j = 0; j < input.cols(); ++j) {
            result[i][j] = std::exp(input[i][j] - max_val);
            sum += result[i][j];
        }

        for (size_t j = 0; j < input.cols(); ++j) {
            result[i][j] /= sum;
        }
    }

    return result;
}


inline Matrix<double> xavier_uniform_init(
    const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / (_input_size + _output_size));
    std::uniform_real_distribution<> dist(-limit, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> xavier_normal_init(
    const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / (_input_size + _output_size));
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> he_normal_init(
    const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / _input_size);
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> he_uniform_init(
    const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / _input_size);
    std::normal_distribution<> dist(-limit, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}


inline Matrix<double> random_init(
    const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    std::uniform_real_distribution<> dist(0.0, 1.0);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> he_normal_init(
    const size_t& _rows,
    const size_t& _cols,
    const size_t& _fan_in,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / static_cast<double>(_fan_in));
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_rows, _cols, 0.0);

    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c)
            m[r][c] = dist(_gen);

    return m;
}


inline Matrix<double> he_uniform_init(
    const size_t& _rows,
    const size_t& _cols,
    const size_t& _fan_in,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / static_cast<double>(_fan_in));
    std::uniform_real_distribution<> dist(-limit, limit);

    Matrix<double> m(_rows, _cols, 0.0);

    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c)
            m[r][c] = dist(_gen);

    return m;
}


inline Matrix<double> xavier_normal_init(
    const size_t& _rows,
    const size_t& _cols,
    const size_t& _fan_in,
    const size_t& _fan_out,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / static_cast<double>(_fan_in + _fan_out));
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_rows, _cols, 0.0);

    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> xavier_uniform_init(
    const size_t& _rows,
    const size_t& _cols,
    const size_t& _fan_in,
    const size_t& _fan_out,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / static_cast<double>(_fan_in + _fan_out));
    std::uniform_real_distribution<> dist(-limit, limit);

    Matrix<double> m(_rows, _cols, 0.0);

    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c)
            m[r][c] = dist(_gen);

    return m;
}

inline Matrix<double> random_init(
    const size_t& _rows,
    const size_t& _cols,
    const double& _begin,
    const double& _end,
    std::mt19937& _gen) {

    std::uniform_real_distribution<> dist(_begin, _end);

    Matrix<double> m(_rows, _cols, 0.0);

    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> softmax(const Matrix<double>& m) {
    const auto ROWS = m.rows();

    Matrix<double> result(ROWS, 1);

    double max_val = m[0][0];
    for (size_t i = 1; i < ROWS; ++i)
        if (m[i][0] > max_val) max_val = m[i][0];

    double sum_exp = 0.0;
    for (size_t i = 0; i < ROWS; ++i) {
        result[i][0] = std::exp(m[i][0] - max_val);
        sum_exp += result[i][0];
    }

    for (size_t i = 0; i < ROWS; ++i)
        result[i][0] /= sum_exp;

    return result;
}

inline double cross_entropy_loss(
    const Matrix<double>& _y_predict,
    const Matrix<double>& _y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < _y_predict.rows(); ++i)
        if (_y_true[i][0] == 1.0)
            loss += -std::log(_y_predict[i][0] + 1e-12);
    return loss;
}

