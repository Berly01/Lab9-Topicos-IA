#ifndef MLP_HPP
#define MLP_HPP

#pragma once
#include <functional>
#include <fstream>
#include <sstream>
#include <tuple>
#include <numeric>
#include <iomanip>
#include "MLP_Layer.hpp"
#include "utils_cuda.hpp"
#include "functions.hpp"


class MLP {

public:
    struct MLPHyperparameters {
        std::vector<size_t> layers = { 2, 2, 3 };
        Initializer initializer = Initializer::RANDOM;
        Optimizer optimizer = Optimizer::NONE;
        Activation activation_func = Activation::RELU;
        double learning_rate = 0.01;
        size_t batch = 16;
        size_t epochs = 10;
        double decay_rate = 0.9;
        double epsilon = 1e-8;
        double weight_decay = 0.01;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double bias_init = 0.0;
        size_t timestep = 1;
        size_t print_every = 10;
        double dropout_rate = 0.2;
        bool shuffle = false;
        bool debug = false;
        bool dropout = false;
        bool save_measure = false;
        bool verbose = false;
    };

private:
    std::vector<MLP_Layer> hidden_layers;
    std::mt19937 gen;
    MLPHyperparameters h;
    bool training_mode = true;

public:

    explicit MLP(const MLPHyperparameters& _h) : gen(std::random_device{}()), h(_h) {

        const auto LAYERS_SIZE = h.layers.size();
        const auto& layers = h.layers;

        for (size_t i = 1; i < LAYERS_SIZE; ++i)
            hidden_layers.emplace_back(layers[i - 1], layers[i], h.initializer, h.bias_init, gen);
    }

    explicit MLP(const std::string& _file_name, const MLPHyperparameters& _h)
        : gen(std::random_device{}()), h(_h) {
            
        std::ifstream in(_file_name + ".dat", std::ios::binary);

        size_t size{};
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        hidden_layers.reserve(size);
      
        for (size_t i = 0; i < size; ++i) {      
            MLP_Layer layer;
            in >> layer.weights >> layer.biases;
            hidden_layers.emplace_back(layer);
        }

        in.close();
    }

    ~MLP() = default;

    TrainingMetrics train(const std::vector<Matrix<double>>& _training_data_x,
        const std::vector<Matrix<double>>& _training_data_y,
        const std::vector<Matrix<double>>& _testing_data_x,
        const std::vector<Matrix<double>>& _testing_data_y) {

        if (_training_data_x.size() != _training_data_y.size()
            || _testing_data_x.size() != _testing_data_y.size())
            throw std::invalid_argument("E");
                
        TrainingMetrics metrics;

        auto get_random_name = [](std::mt19937& _gen, const size_t& _len) {
            std::stringstream ss;
            std::uniform_int_distribution<> dist(48, 57);

            for (size_t i = 0; i < _len; ++i)
                ss << static_cast<char>(dist(_gen));

            return ss.str();
            };

        const auto TRAINING_SIZE = _training_data_x.size();
        const auto LAYERS_SIZE = hidden_layers.size();

        const auto LOSS_FILE_NAME = get_random_name(gen, 6) + "-mlp-loss.csv";
        const auto ACCURACY_FILE_NAME = get_random_name(gen, 6) + "-mlp-accuracy.csv";
        double total_loss = 0;

        std::vector<size_t> random_indices(TRAINING_SIZE);
        std::iota(random_indices.begin(), random_indices.end(), 0);

        size_t begin{};
        size_t end{};

        if (h.save_measure) save_measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, -1, -1, -1, true);

        for (size_t e = 0; e < h.epochs; ++e) {

            training_mode = true;

            if (h.shuffle) std::shuffle(random_indices.begin(), random_indices.end(), gen);

            for (begin = 0; begin < TRAINING_SIZE; begin += h.batch) {
                end = std::min(begin + h.batch, TRAINING_SIZE);
                std::vector<Matrix<double>> batch_x, batch_y;
                for (size_t j = begin; j < end; ++j) {
                    batch_x.push_back(_training_data_x[random_indices[j]]);
                    batch_y.push_back(_training_data_y[random_indices[j]]);
                }
                total_loss += update_mini_batch(batch_x, batch_y);
            }

            total_loss /= static_cast<double>(TRAINING_SIZE);

            const auto accuracy = get_accuracy(_testing_data_x, _testing_data_y);

            metrics.train_losses.push_back(total_loss);
            metrics.train_accuracies.push_back(accuracy);

            if (h.verbose && (e + 1) % h.print_every == 0) {
                metrics.print_epoch_metrics(e + 1, total_loss, accuracy);
            }

            if (h.save_measure) save_measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, e, total_loss, accuracy, false);

            total_loss = 0;
        }

        if (h.verbose) {
            std::cout << "Entrenamiento Finalizado" << std::endl
                << "Perdida Final: "
                << metrics.train_losses.back() << std::endl
                << "Presicion Final: " 
                << metrics.train_accuracies.back() * 100 << "%" << std::endl;          
        }

        return metrics;
    }

    double get_accuracy(const std::vector<Matrix<double>>& _testing_data_x,
        const std::vector<Matrix<double>>& _testing_data_y) {

        if (training_mode) training_mode = false;

        int correct = 0;
        const auto TESTING_SIZE = _testing_data_x.size();

        for (size_t i = 0; i < TESTING_SIZE; ++i) {

            auto y_pre = predict(_testing_data_x[i]);

            size_t predicted = 0;
            double max_prob = y_pre[0][0];
            for (size_t j = 1; j < y_pre.rows(); ++j) {
                if (y_pre[j][0] > max_prob) {
                    max_prob = y_pre[j][0];
                    predicted = j;
                }
            }

            size_t actual = 0;
            for (size_t j = 0; j < _testing_data_y[i].rows(); ++j) {
                if (_testing_data_y[i][j][0] == 1.0) {
                    actual = j;
                    break;
                }
            }

            if (predicted == actual)
                ++correct;
        }

        return static_cast<double>(correct) / static_cast<double>(TESTING_SIZE);
    }

    Matrix<double> predict(const Matrix<double>& x) {
        return forward(x);
    }

    void save_weights(const std::string& _file_name) {
        std::ofstream out(_file_name + ".dat", std::ofstream::trunc | std::ios::binary);
        size_t size = hidden_layers.size();

        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& l : hidden_layers) 
            out << l.weights << l.biases;                        
        out.close();
    }

private:
    Matrix<double> forward(const Matrix<double>& _input) {
        Matrix<double> x = _input;
        std::uniform_real_distribution<> dropout_dist(0.0, 1.0);

        auto begin = hidden_layers.begin();
        auto end = hidden_layers.end();

        for (; begin != end; ++begin) {
            auto& layer = *begin;
            layer.z = CU::basic_oper(CU::dot_product(layer.weights, x), layer.biases, Operation::ADD);

            if (begin != end - 1) {
                layer.activation = activacion(layer.z, h.activation_func);
           
                if (h.dropout) {
                    if (training_mode) {
                        for (size_t r = 0; r < layer.activation.rows(); ++r) {
                            double keep = (dropout_dist(gen) > h.dropout) ? 1.0 : 0.0;
                            layer.mask[r][0] = keep;
                            layer.activation[r][0] *= keep;
                        }
                    }
                    else {
                        for (size_t r = 0; r < layer.activation.rows(); ++r)
                            layer.activation[r][0] *= (1.0 - h.dropout);
                    }
                }
            }             
            else {
                layer.activation = softmax(layer.z);
            }
                
            x = layer.activation;
        }

        return x;
    }
      
    auto backward(const Matrix<double>& _x, const Matrix<double>& _y) {

        std::vector<Matrix<double>> nabla_w(hidden_layers.size());
        std::vector<Matrix<double>> nabla_b(hidden_layers.size());

        Matrix<double> delta_i = CU::basic_oper(hidden_layers.back().activation, _y, Operation::SUBT);

        nabla_w.back() = CU::outer_product(delta_i, hidden_layers[hidden_layers.size() - 2].activation);
        nabla_b.back() = delta_i;
        const auto LAYERS = static_cast<int>(hidden_layers.size() - 2);

        for (int l = LAYERS; l >= 0; --l) {
            Matrix<double> sp = activacion_deri(hidden_layers[l].z, h.activation_func);       
            Matrix<double> wTp = CU::dot_product(hidden_layers[l + 1].weights.transpose(), delta_i);

            for (size_t i = 0; i < wTp.rows(); ++i)
                wTp[i][0] *= sp[i][0];

            delta_i = wTp;
            Matrix<double> prev_activation = (l == 0) ? _x : hidden_layers[l - 1].activation;
            nabla_b[l] = delta_i;
            nabla_w[l] = CU::outer_product(delta_i, prev_activation);
        }

        return std::make_tuple(nabla_w, nabla_b);
    }

    double update_mini_batch(
        const std::vector<Matrix<double>>& _batch_x,
        const std::vector<Matrix<double>>& _batch_y) {

        const auto LAYERS = hidden_layers.size();
        const auto BATCH = _batch_x.size();
        double total_loss = 0;

        std::vector<Matrix<double>> nabla_w_acc(LAYERS);
        std::vector<Matrix<double>> nabla_b_acc(LAYERS);

        for (size_t i = 0; i < LAYERS; ++i) {
            nabla_w_acc[i] = Matrix<double>(hidden_layers[i].weights.rows(), hidden_layers[i].weights.cols(), 0.0);
            nabla_b_acc[i] = Matrix<double>(hidden_layers[i].biases.rows(), 1, 0.0);
        }

        for (size_t i = 0; i < BATCH; ++i) {
            total_loss += cross_entropy_loss(forward(_batch_x[i]), _batch_y[i]);
            const auto tuple = backward(_batch_x[i], _batch_y[i]);
            auto& delta_nabla_w = std::get<0>(tuple);
            auto& delta_nabla_b = std::get<1>(tuple);

            for (size_t j = 0; j < LAYERS; ++j) {
                nabla_w_acc[j] = nabla_w_acc[j] + delta_nabla_w[j];
                nabla_b_acc[j] = nabla_b_acc[j] + delta_nabla_b[j];
            }
        }

        optimizate(nabla_w_acc, nabla_b_acc, BATCH);

        return total_loss;
    }

    void optimizate(
        const std::vector<Matrix<double>>& _nabla_w_acc,
        const std::vector<Matrix<double>>& _nabla_b_acc,
        const size_t& _batch) {

        const auto LAYERS = hidden_layers.size();

        if (h.optimizer == Optimizer::RMS_PROP) {
            const double LR = h.learning_rate / static_cast<double>(_batch);
            SquareFunctor square_f;

            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];

                layer.cache_w = CU::basic_oper(CU::scalar_product(layer.cache_w, h.decay_rate), CU::scalar_product(CU::apply_function(_nabla_w_acc[i], square_f), (1.0 - h.decay_rate)), Operation::ADD);
                layer.cache_b = CU::basic_oper(CU::scalar_product(layer.cache_b, h.decay_rate), CU::scalar_product(CU::apply_function(_nabla_b_acc[i], square_f), (1.0 - h.decay_rate)), Operation::ADD);

                Matrix<double> adjusted_w = _nabla_w_acc[i];
                Matrix<double> adjusted_b = _nabla_b_acc[i];

                for (size_t r = 0; r < adjusted_w.rows(); ++r)
                    for (size_t c = 0; c < adjusted_w.cols(); ++c)
                        adjusted_w[r][c] /= std::sqrt(layer.cache_w[r][c] + h.epsilon);

                for (size_t r = 0; r < adjusted_b.rows(); ++r)
                    adjusted_b[r][0] /= std::sqrt(layer.cache_b[r][0] + h.epsilon);

                layer.weights = CU::basic_oper(layer.weights, CU::scalar_product(adjusted_w, LR), Operation::SUBT);
                layer.biases = CU::basic_oper(layer.biases, CU::scalar_product(adjusted_b, LR), Operation::SUBT);
            }
        }
        else if (h.optimizer == Optimizer::ADAM) {
            SquareFunctor square_f;
            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];

                layer.m_w = CU::basic_oper(CU::scalar_product(layer.m_w, h.beta1), CU::scalar_product(_nabla_w_acc[i], (1.0 - h.beta1)), Operation::ADD);
                layer.v_w = CU::basic_oper(CU::scalar_product(layer.v_w, h.beta2), CU::scalar_product(CU::apply_function(_nabla_w_acc[i], square_f), (1.0 - h.beta2)), Operation::ADD);

                layer.m_b = CU::basic_oper(CU::scalar_product(layer.m_b, h.beta1), CU::scalar_product(_nabla_b_acc[i], (1.0 - h.beta1)), Operation::ADD);
                layer.v_b = CU::basic_oper(CU::scalar_product(layer.v_b, h.beta2), CU::scalar_product(CU::apply_function(_nabla_b_acc[i], square_f), (1.0 - h.beta2)), Operation::ADD);

                const double LR_T = h.learning_rate * std::sqrt(1.0 - std::pow(h.beta2, h.timestep)) / (1.0 - std::pow(h.beta1, h.timestep));

                for (size_t r = 0; r < layer.weights.rows(); ++r) {
                    for (size_t c = 0; c < layer.weights.cols(); ++c) {
                        double m_hat = layer.m_w[r][c];
                        double v_hat = layer.v_w[r][c];
                        double update = LR_T * m_hat / (std::sqrt(v_hat) + h.epsilon);
                        update += h.weight_decay * layer.weights[r][c];
                        layer.weights[r][c] -= update;
                    }
                }

                for (size_t r = 0; r < layer.biases.rows(); ++r) {
                    double m_hat = layer.m_b[r][0];
                    double v_hat = layer.v_b[r][0];
                    double update = LR_T * m_hat / (std::sqrt(v_hat) + h.epsilon);
                    layer.biases[r][0] -= update;
                }
            }
            ++h.timestep;
        }
        else {
            const double LR = h.learning_rate / static_cast<double>(h.batch);
            for (size_t i = 0; i < LAYERS; ++i) {
                auto& layer = hidden_layers[i];
                layer.weights = CU::basic_oper(layer.weights, CU::scalar_product(_nabla_w_acc[i], LR), Operation::SUBT);
                layer.biases = CU::basic_oper(layer.biases, CU::scalar_product(_nabla_b_acc[i], LR), Operation::SUBT);
            }
        }
    }

    void save_measure(
        const std::string& _loss_file_path,
        const std::string& _accuracy_file_path,
        const size_t& _EPOCH,
        const double& _LOSS,
        const double& _ACCURACY,
        const bool& _headers) {       

        if (_headers) {
            std::ofstream loss_file(_loss_file_path, std::ofstream::trunc);
            std::ofstream accuracy_file(_accuracy_file_path, std::ofstream::trunc);

            loss_file << "epoca,perdida\n";
            accuracy_file << "epoca,precision\n";

            loss_file.close();
            accuracy_file.close();
        }
        else {
            std::ofstream loss_file(_loss_file_path, std::ofstream::app);
            std::ofstream accuracy_file(_accuracy_file_path, std::ofstream::app);

            loss_file << _EPOCH << ',' << _LOSS << '\n';
            accuracy_file << _EPOCH << ',' << _ACCURACY << '\n';

            loss_file.close();
            accuracy_file.close();
        }
    }

    Matrix<double> activacion(const Matrix<double>& _m, const Activation& _f) const {
        if (_f == Activation::RELU)
            return CU::apply_function(_m, ReluFunctor());
        else if (_f == Activation::SIGMOID)
            return CU::apply_function(_m, SigmoidFunctor());
        else
            return CU::apply_function(_m, TanhFunctor());
    }

    Matrix<double> activacion_deri(const Matrix<double>& _m, const Activation& _f) const {

        if (_f == Activation::RELU)
            return CU::apply_function(_m, DReluFunctor());
        else if (_f == Activation::SIGMOID)
            return CU::apply_function(_m, DSigmoidFunctor());
        else
            return CU::apply_function(_m, DTanhFunctor());
    }

};

#endif