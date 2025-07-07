#pragma once
#include "Matrix.hpp"
#include <array>

class DataLoader {
private:

public:

	~DataLoader() = default;
  
    static auto cnn_load_one_channel(const std::string& _dataset_name, const std::string& _split) {

        std::string image_path;
        std::string label_path;

        std::vector<uint8_t> labels;
        std::vector<uint8_t> images;

        size_t data_size;
        size_t image_size;

        std::vector<Matrix<double>> n_labels;
        std::vector<std::vector<Matrix<double>>> n_images;


        if (_dataset_name == "minist") {
            image_size = 28 * 28;
            if (_split == "train") {
                image_path = "mnist/train-images-idx3-ubyte";
                label_path = "mnist/train-labels-idx1-ubyte";
                data_size = 60000;
            }
            else {
                image_path = "mnist/t10k-images-idx3-ubyte";
                label_path = "mnist/t10k-labels-idx1-ubyte";
                data_size = 10000;
            }
        }
        else {
            image_size = 28 * 28;
            if (_split == "train") {
                image_path = "fashion/train-images-idx3-ubyte";
                label_path = "fashion/train-labels-idx1-ubyte";
                data_size = 60000;
            }
            else {
                image_path = "fashion/t10k-images-idx3-ubyte";
                label_path = "fashion/t10k-labels-idx1-ubyte";
                data_size = 10000;
            }
        }

        labels = read_ubyte_label_file(label_path);
        images = read_ubyte_image_file(image_path);
        n_labels.reserve(data_size);
        n_images.reserve(data_size);

        for (size_t i = 0; i < data_size; ++i) {
            auto m = Matrix<double>(10, 1, 0.0);
            m[static_cast<size_t>(labels[i])][0] = 1.0;
            n_labels.emplace_back(m);
        }
            
        for (size_t i = 0; i < data_size; ++i) {
            std::vector<double> vec(image_size, 0.0);
            
            for (size_t j = 0; j < image_size; ++j) 
                vec[j] = static_cast<double>(images[j + (i * image_size)]) / 255.0;
                                  
            std::vector<Matrix<double>> v = { Matrix<double>(vec) };
       
            n_images.emplace_back(v);
        }
            
        return std::make_tuple(n_labels, n_images);
	}

    static auto mlp_load_one_channel(const std::string& _dataset_name, const std::string& _split) {

        std::string image_path;
        std::string label_path;

        std::vector<uint8_t> labels;
        std::vector<uint8_t> images;

        size_t data_size;
        size_t image_size;

        std::vector<Matrix<double>> n_labels;
        std::vector<Matrix<double>> n_images;


        if (_dataset_name == "minist") {
            image_size = 28 * 28;
            if (_split == "train") {
                image_path = "mnist/train-images-idx3-ubyte";
                label_path = "mnist/train-labels-idx1-ubyte";
                data_size = 60000;
            }
            else {
                image_path = "mnist/t10k-images-idx3-ubyte";
                label_path = "mnist/t10k-labels-idx1-ubyte";
                data_size = 10000;
            }
        }
        else {
            image_size = 28 * 28;
            if (_split == "train") {
                image_path = "fashion/train-images-idx3-ubyte";
                label_path = "fashion/train-labels-idx1-ubyte";
                data_size = 60000;
            }
            else {
                image_path = "fashion/t10k-images-idx3-ubyte";
                label_path = "fashion/t10k-labels-idx1-ubyte";
                data_size = 10000;
            }
        }

        labels = read_ubyte_label_file(label_path);
        images = read_ubyte_image_file(image_path);
        n_labels.reserve(data_size);
        n_images.reserve(data_size);

        for (size_t i = 0; i < data_size; ++i) {
            auto m = Matrix<double>(10, 1, 0.0);
            m[static_cast<size_t>(labels[i])][0] = 1.0;
            n_labels.emplace_back(m);
        }

        for (size_t i = 0; i < data_size; ++i) {
            std::vector<double> vec(image_size, 0.0);

            for (size_t j = 0; j < image_size; ++j)
                vec[j] = static_cast<double>(images[j + (i * image_size)]) / 255.0;

            n_images.emplace_back(vec);
        }

        return std::make_tuple(n_labels, n_images);
    }


private:
	DataLoader() = default;
	
    static uint32_t read_big_endian_uint32(std::ifstream& stream) {
        std::array<uint8_t, 4> bytes;
        stream.read(reinterpret_cast<char*>(bytes.data()), 4);
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    }

    static std::vector<uint8_t> read_ubyte_image_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("No se pudo abrir el archivo: " + filename);

        uint32_t magic_number = read_big_endian_uint32(file);
        uint32_t num_images = read_big_endian_uint32(file);
        uint32_t rows = read_big_endian_uint32(file);
        uint32_t cols = read_big_endian_uint32(file);

        size_t image_size = rows * cols;
        std::vector<uint8_t> buffer(num_images * image_size);
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        if (!file)
            throw std::runtime_error("Error al leer los datos del archivo.");

        return buffer;
    }


    static std::vector<uint8_t> read_ubyte_label_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("No se pudo abrir el archivo: " + filename);

        uint32_t magic_number = read_big_endian_uint32(file);
        uint32_t num_labels = read_big_endian_uint32(file);

        std::vector<uint8_t> buffer(num_labels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

        if (!file)
            throw std::runtime_error("Error al leer los datos del archivo.");


        return buffer;
    }
};

