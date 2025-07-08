#pragma once
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include <numeric>
#include <sstream>
#include <iomanip>

class CNN {
private:
    std::vector<ConvLayer> conv_layers_;
    std::vector<DenseLayer> dense_layers_;
    std::vector<std::vector<Matrix<double>>> conv_outputs_;
    std::vector<Matrix<double>> dense_outputs_;

public:
    struct CNNHyperparameters {
        size_t epochs = 100;
        size_t batch_size = 32;
        double learning_rate = 0.001;
        bool shuffle_data = true;
        bool verbose = true;
        size_t print_every = 10;
        bool early_stopping = false;
        double patience = 10;
        double min_delta = 0.001;
        bool save_measure = false;
    };

    CNN(const std::vector<ConvLayer>& conv_layers, const std::vector<DenseLayer>& dense_layers)
        : conv_layers_(conv_layers), dense_layers_(dense_layers) {
    }

    Matrix<double> forward(const std::vector<Matrix<double>>& input) {
        std::vector<Matrix<double>> current = input;

        conv_outputs_.clear();
        conv_outputs_.push_back(current);

        for (auto& layer : conv_layers_) {
            current = layer.forward(current);
            conv_outputs_.push_back(current);
        }

        // Flatten la salida de las capas convolucionales
        Matrix<double> flattened = flatten(current);

        dense_outputs_.clear();
        dense_outputs_.push_back(flattened);

        Matrix<double> dense_output = flattened;
        for (auto& layer : dense_layers_) {
            dense_output = layer.forward(dense_output);
            dense_outputs_.push_back(dense_output);
        }

        return dense_output;
    }

    Matrix<double> predict(const std::vector<Matrix<double>>& input) {
        Matrix<double> output = forward(input);
        return softmax(output);
    }
    
    std::vector<Matrix<double>> backward(const Matrix<double>& grad_output) {
        Matrix<double> grad = grad_output;

        for (int i = dense_layers_.size() - 1; i >= 0; --i) {
            grad = dense_layers_[i].backward(grad);
        }

        // Convertir gradiente a formato de canales para capas convolucionales
        std::vector<Matrix<double>> grad_channels = unflatten(grad);

        for (int i = conv_layers_.size() - 1; i >= 0; --i) {
            grad_channels = conv_layers_[i].backward(grad_channels);
        }

        return grad_channels;
    }

    void update_parameters() {
        for (auto& layer : conv_layers_) {
            layer.update_parameters();
        }

        for (auto& layer : dense_layers_) {
            layer.update_parameters();
        }
    }

    TrainingMetrics train(const std::vector<std::vector<Matrix<double>>>& train_data,
        const std::vector<Matrix<double>>& train_labels,
        const std::vector<std::vector<Matrix<double>>>& val_data,
        const std::vector<Matrix<double>>& val_labels,
        const CNNHyperparameters& params) {

        if (train_data.size() != train_labels.size()
            || val_data.size() != val_labels.size())
            throw std::invalid_argument("E");
        

        TrainingMetrics metrics;
        std::vector<size_t> indices(train_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        auto get_random_name = [](std::mt19937& _gen, const size_t& _len) {
            std::stringstream ss;
            std::uniform_int_distribution<> dist(48, 57);

            for (size_t i = 0; i < _len; ++i)
                ss << static_cast<char>(dist(_gen));

            return ss.str();
            };

        const auto LOSS_FILE_NAME = get_random_name(gen, 6) + "-cnn-loss.csv";
        const auto ACCURACY_FILE_NAME = get_random_name(gen, 6) + "-cnn-accuracy.csv";

        if (params.save_measure) save_measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, -1, -1, -1, true);

        if (params.verbose) {
            std::cout << "Iniciando entrenamiento para " << params.epochs << " epocas" << std::endl;
            std::cout << "Batch: " << params.batch_size << std::endl;
            std::cout << "Taza de aprendizaje: " << params.learning_rate << std::endl;
        }

        for (size_t epoch = 0; epoch < params.epochs; ++epoch) {

            if (params.shuffle_data) {
                std::shuffle(indices.begin(), indices.end(), gen);
            }

            double epoch_train_loss = 0.0;
            double epoch_train_accuracy = 0.0;
            size_t num_batches = 0;

            for (size_t batch_start = 0; batch_start < train_data.size(); batch_start += params.batch_size) {
                size_t batch_end = std::min(batch_start + params.batch_size, train_data.size());
                size_t actual_batch_size = batch_end - batch_start;

                double batch_loss = 0.0;

                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t idx = indices[i];

                    Matrix<double> output = forward(train_data[idx]);

                    Matrix<double> predictions = softmax(output);

                    double sample_loss = cross_entropy_loss(predictions, train_labels[idx]);
                    batch_loss += sample_loss;

                    Matrix<double> grad_output = predictions - train_labels[idx];

                    backward(grad_output);
                }

                update_parameters();

                batch_loss /= actual_batch_size;
                epoch_train_loss += batch_loss;
                num_batches++;
            }

            epoch_train_loss /= num_batches;

            epoch_train_accuracy = get_accuracy(val_data, val_labels);

            metrics.train_losses.push_back(epoch_train_loss);
            metrics.train_accuracies.push_back(epoch_train_accuracy);


            if (params.verbose && (epoch + 1) % params.print_every == 0) {
                metrics.print_epoch_metrics(epoch + 1, epoch_train_loss, epoch_train_accuracy);
            }

            if (params.save_measure) save_measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, epoch + 1, epoch_train_loss, epoch_train_accuracy, false);
        }

        if (params.verbose) {
            std::cout << "Entrenamiento Finalizado" << std::endl
                << "Perdida Final: "
                << metrics.train_losses.back() << std::endl
                << "Presicion Final: "
                << metrics.train_accuracies.back() * 100 << "%" << std::endl;          
        }

        return metrics;
    }


    static size_t calculate_dense_layer_inputs(size_t initial_height,
                                  size_t initial_width,
                                  const std::vector<ConvLayer>& conv_layers) {
        if (conv_layers.empty()) {
            throw std::invalid_argument("Se requiere al menos una capa convolucional");
        }
        
        size_t current_height = initial_height;
        size_t current_width = initial_width;
        size_t current_channels = 1; // Asumiendo entrada inicial de 1 canal
        
        for (const auto& layer : conv_layers) {
            size_t kernel_size = layer.get_kernel_size();
            size_t padding = layer.get_padding_size();
            size_t stride = layer.get_stride_size();
            size_t pool_size = layer.get_pool_size();
            bool uses_pooling = layer.uses_pooling();
            
            // Aplicar convolución: (input + 2*padding - kernel) / stride + 1
            current_height = (current_height + 2 * padding - kernel_size) / stride + 1;
            current_width = (current_width + 2 * padding - kernel_size) / stride + 1;
            
            // Aplicar pooling si está habilitado
            if (uses_pooling && pool_size > 0) {
                current_height = current_height / pool_size;
                current_width = current_width / pool_size;
            }
            
            // Actualizar número de canales de salida
            current_channels = layer.get_out_channels();
            
            if (current_height == 0 || current_width == 0) {
                throw std::runtime_error("Las dimensiones se redujeron a cero");
            }
        }
        
        return current_height * current_width * current_channels;
}


private:

    Matrix<double> flatten(const std::vector<Matrix<double>>& channels) {
        std::vector<double> flatten_output;
        for (const auto& channel : channels) {
            const auto flat = channel.flat();
            flatten_output.insert(flatten_output.end(), flat.cbegin(), flat.cend());
        }
        return Matrix<double>(flatten_output);
    }

    std::vector<Matrix<double>> unflatten(const Matrix<double>& flattened) {

        const auto& last_conv_output = conv_outputs_.back();
        std::vector<Matrix<double>> result;

        const auto flat_data = flattened.flat();
        size_t idx = 0;

        for (const auto& channel : last_conv_output) {
            Matrix<double> channel_grad(channel.rows(), channel.cols());
            for (size_t i = 0; i < channel.rows(); ++i) {
                for (size_t j = 0; j < channel.cols(); ++j) {
                    if (idx < flat_data.size()) {
                        channel_grad[i][j] = flat_data[idx++];
                    }
                }
            }
            result.push_back(channel_grad);
        }

        return result;
    }

    double get_accuracy(const std::vector<std::vector<Matrix<double>>>& _testing_data_x,
        const std::vector<Matrix<double>>& _testing_data_y) {

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

};