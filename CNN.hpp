#pragma once
#include "ConvLayer.hpp"
#include "DenseLayer.hpp"
#include <numeric>
#include <sstream>
#include <iomanip>

struct TrainingMetrics {
    std::vector<double> train_losses;
    std::vector<double> train_accuracies;
    std::vector<double> val_losses;
    std::vector<double> val_accuracies;

    void print_epoch_metrics(size_t epoch, double train_loss, double train_acc,
        double val_loss, double val_acc) const {
        std::cout << "Epoch " << std::setw(3) << epoch
            << " | Train Loss: " << train_loss
            << " | Train Acc: " << train_acc * 100 << "%"
            << " | Val Loss: " << val_loss
            << " | Val Acc: " << val_acc * 100 << "%"
            << std::endl;
    }
};

class CNN {

public:
    struct CNNHyperparameters {
        size_t epochs = 100;
        size_t batch_size = 32;
        double learning_rate = 0.001;
        bool shuffle_data = true;
        bool verbose = true;
        bool save_measure = false;
        size_t print_every = 10;
        bool early_stopping = false;
        double patience = 10;
        double min_delta = 0.001;
    };

    CNN(const std::vector<ConvLayer>& conv_layers, const std::vector<DenseLayer>& dense_layers);

    Matrix<double> forward(const std::vector<Matrix<double>>& input);

    Matrix<double> predict(const std::vector<Matrix<double>>& input);

    std::vector<Matrix<double>> backward(const Matrix<double>& grad_output);

    void update_parameters();

    TrainingMetrics train(
        const std::vector<std::vector<Matrix<double>>>& train_data,
        const std::vector<Matrix<double>>& train_labels,
        const std::vector<std::vector<Matrix<double>>>& val_data,
        const std::vector<Matrix<double>>& val_labels,
        const CNNHyperparameters& params);

private:
    Matrix<double> flatten(const std::vector<Matrix<double>>& channels);

    std::vector<Matrix<double>> unflatten(const Matrix<double>& flattened);

    void measure(
        const std::string& _loss_file_path,
        const std::string& _accuracy_file_path,
        const size_t& _EPOCH,
        const double& _LOSS,
        const double& _ACCURACY,
        const bool& _headers);

    std::vector<ConvLayer> conv_layers_;
    std::vector<DenseLayer> dense_layers_;
    std::vector<std::vector<Matrix<double>>> conv_outputs_;
    std::vector<Matrix<double>> dense_outputs_;
};

CNN::CNN(const std::vector<ConvLayer>& conv_layers, const std::vector<DenseLayer>& dense_layers)
    : conv_layers_(conv_layers), dense_layers_(dense_layers) {
}

Matrix<double> CNN::forward(const std::vector<Matrix<double>>& input) {

    std::vector<Matrix<double>> current = input;

    conv_outputs_.clear();
    conv_outputs_.push_back(current);

    for (auto& layer : conv_layers_) {
        current = layer.forward(current);
        conv_outputs_.push_back(current);
    }

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

Matrix<double> CNN::predict(const std::vector<Matrix<double>>& input) {
    Matrix<double> output = forward(input);
    return softmax(output);
}

std::vector<Matrix<double>> CNN::backward(const Matrix<double>& grad_output) {

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

void CNN::update_parameters() {

    for (auto& layer : conv_layers_) {
        layer.update_parameters();
    }

    for (auto& layer : dense_layers_) {
        layer.update_parameters();
    }
}

TrainingMetrics CNN::train(const std::vector<std::vector<Matrix<double>>>& train_data,
    const std::vector<Matrix<double>>& train_labels,
    const std::vector<std::vector<Matrix<double>>>& val_data,
    const std::vector<Matrix<double>>& val_labels,
    const CNNHyperparameters& params = CNNHyperparameters{}) {

    TrainingMetrics metrics;
    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (params.verbose) {
        std::cout << "Entranamiento para " << params.epochs << " epocas" << std::endl;
        std::cout << "Batch: " << params.batch_size << std::endl;
        std::cout << "Taza de aprendizaje: " << params.learning_rate << std::endl;
    }

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

    if (params.save_measure) measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, -1, -1, -1, true);

 
    for (size_t epoch = 1; epoch <= params.epochs; ++epoch) {

        std::cout << "Epoca " << epoch << '\n';

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
            double batch_accuracy = 0.0;

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

            epoch_train_loss += batch_loss / static_cast<double>(actual_batch_size);
        }
     
        std::cout << "Validacion\n";
        double val_loss = -1;
        double val_accuracy = -1;
        int correct = 0;

        for (size_t i = 0; i < val_data.size(); ++i) {
            Matrix<double> output = forward(val_data[i]);
            Matrix<double> prediction = softmax(output);

            size_t predicted = 0;
            double max_prob = prediction[0][0];
            for (size_t j = 1; j < prediction.rows(); ++j) {
                if (prediction[j][0] > max_prob) {
                    max_prob = prediction[j][0];
                    predicted = j;
                }
            }

            size_t actual = 0;
            for (size_t j = 0; j < val_labels[i].rows(); ++j) {
                if (val_labels[i][j][0] == 1.0) {
                    actual = j;
                    break;
                }
            }

            if (predicted == actual)
                ++correct;
        }

        epoch_train_accuracy = static_cast<double>(correct) / static_cast<double>(val_data.size());

        metrics.train_losses.push_back(epoch_train_loss);
        metrics.train_accuracies.push_back(epoch_train_accuracy);
        metrics.val_losses.push_back(val_loss);
        metrics.val_accuracies.push_back(val_accuracy);

        if (params.verbose && (epoch) % params.print_every == 0) {
            metrics.print_epoch_metrics(epoch, epoch_train_loss, epoch_train_accuracy, val_loss, val_accuracy);
        }

        if (params.save_measure) measure(LOSS_FILE_NAME, ACCURACY_FILE_NAME, epoch, epoch_train_loss, epoch_train_accuracy, false);
    }

    if (params.verbose) {
        std::cout << "Entrenamiento Finalizado" << std::endl;
        std::cout << "Final Presicion Final: "
            << metrics.val_accuracies.back() * 100 << "%" << std::endl;
    }

    return metrics;
}

Matrix<double> CNN::flatten(const std::vector<Matrix<double>>& channels) {
    std::vector<double> flatten_output;
    for (const auto& channel : channels) {
        const auto flat = channel.flat();
        flatten_output.insert(flatten_output.end(), flat.cbegin(), flat.cend());
    }
    return Matrix<double>(flatten_output);
}

std::vector<Matrix<double>> CNN::unflatten(const Matrix<double>& flattened) {
    
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


void CNN::measure(
    const std::string& _loss_file_path,
    const std::string& _accuracy_file_path,
    const size_t& _EPOCH,
    const double& _LOSS,
    const double& _ACCURACY,
    const bool& _headers) {

    std::ofstream accuracy_file(_accuracy_file_path, std::ofstream::trunc);
    std::ofstream loss_file(_loss_file_path, std::ofstream::trunc);

    if (_headers) {
        loss_file << "epoca,perdida\n";
        accuracy_file << "epoca,precision\n";
    }
    else {
        loss_file << _EPOCH << ',' << _LOSS << '\n';
        accuracy_file << _EPOCH << ',' << _ACCURACY << '\n';
    }
}