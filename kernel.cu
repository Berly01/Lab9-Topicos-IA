#include "DataLoader.hpp"
#include "CNN.hpp"
#include "MLP.hpp"

void fashion_cnn() {

    auto TRAIN_FASHION = DataLoader::cnn_load_one_channel("fashion", "train");
    auto TEST_FASHION = DataLoader::cnn_load_one_channel("fashion", "test");

    auto& train_labels_data = std::get<0>(TRAIN_FASHION);
    auto& train_images_data = std::get<1>(TRAIN_FASHION);

    auto& test_labels_data = std::get<0>(TEST_FASHION);
    auto& test_images_data = std::get<1>(TEST_FASHION);
 
    train_labels_data.erase(train_labels_data.begin(), train_labels_data.begin() + 59000);
    train_images_data.erase(train_images_data.begin(), train_images_data.begin() + 59000);

    test_labels_data.erase(test_labels_data.begin(), test_labels_data.begin() + 9500);
    test_images_data.erase(test_images_data.begin(), test_images_data.begin() + 9500);


    ConvLayer::ConvLayerHyperparameters conv_params1;
    conv_params1.padding = true;
    conv_params1.pooling = true;
    conv_params1.activation = true;
    conv_params1.debug = false;
    conv_params1.learning_rate = 0.002;
    conv_params1.padding_size = 1;
    conv_params1.stride = 1;
    conv_params1.pool_size = 2;
    conv_params1.activation_func = Activation::RELU;
    conv_params1.init = Initializer::N_HE;
    conv_params1.pool_mode = PoolMode::MAX;

    ConvLayer::ConvLayerHyperparameters conv_params2;
    conv_params2.padding = true;
    conv_params2.pooling = true;
    conv_params2.activation = true;
    conv_params2.debug = false;
    conv_params2.learning_rate = 0.002;
    conv_params2.padding_size = 1;
    conv_params2.stride = 1;
    conv_params2.pool_size = 2;
    conv_params2.activation_func = Activation::RELU;
    conv_params2.init = Initializer::N_HE;
    conv_params2.pool_mode = PoolMode::MAX;

    std::vector<ConvLayer> conv_layers;
    conv_layers.emplace_back(1, 16, 3, conv_params1);
    conv_layers.emplace_back(16, 4, 3, conv_params2);

    DenseLayer::DenseLayerHyperparameters dense_params;
    dense_params.learning_rate = 0.002;
    dense_params.activation_func = Activation::RELU;
    dense_params.use_bias = true;
    dense_params.init = Initializer::N_HE;

    auto first_dense_layer = 8 * 8 * 4;

    std::vector<DenseLayer> dense_layers;
    dense_layers.emplace_back(first_dense_layer, 16, dense_params);

    DenseLayer::DenseLayerHyperparameters output_params = dense_params;
    output_params.use_activation = false;
    dense_layers.emplace_back(16, 10, output_params);

    CNN cnn(conv_layers, dense_layers);

    CNN::CNNHyperparameters train_params;
    train_params.epochs = 1;
    train_params.batch_size = 32;
    train_params.learning_rate = 0.002;
    train_params.verbose = true;
    train_params.print_every = 1;
    train_params.save_measure = true;

    TrainingMetrics metrics = cnn.train(train_images_data, train_labels_data, test_images_data, test_labels_data, train_params);
}

void fashion_mlp() {

    auto TRAIN_FASHION = DataLoader::mlp_load_one_channel("fashion", "train");
    auto TEST_FASHION = DataLoader::mlp_load_one_channel("fashion", "test");

    auto& train_labels_data = std::get<0>(TRAIN_FASHION);
    auto& train_images_data = std::get<1>(TRAIN_FASHION);

    auto& test_labels_data = std::get<0>(TEST_FASHION);
    auto& test_images_data = std::get<1>(TEST_FASHION);

    train_labels_data.erase(train_labels_data.begin(), train_labels_data.begin() + 59000);
    train_images_data.erase(train_images_data.begin(), train_images_data.begin() + 59000);

    test_labels_data.erase(test_labels_data.begin(), test_labels_data.begin() + 9500);
    test_images_data.erase(test_images_data.begin(), test_images_data.begin() + 9500);
 
    MLP::MLPHyperparameters h;
    h.learning_rate = 0.002;
    h.batch = 32;
    h.epochs = 1;
    h.shuffle = true;
    h.save_measure = true;
    h.layers = { 784, 16, 10 };
    h.initializer = Initializer::N_HE;
    h.optimizer = Optimizer::NONE;
    h.activation_func = Activation::RELU;
    h.verbose = true;
    h.print_every = 1;

    MLP mlp(h);

    mlp.train(train_images_data, train_labels_data, test_images_data, test_labels_data);
}


int main() {

    fashion_cnn();

	return 0;
}
