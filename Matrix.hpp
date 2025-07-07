#ifndef MATRIX_HPP
#define MATRIX_HPP

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <functional>

template <typename T>
class Matrix {
private:
    size_t n_rows{};
    size_t n_cols{};
    std::vector<std::vector<T>> data;

public:

    Matrix() = default;

    Matrix(const Matrix& other) = default;

    explicit Matrix(const size_t& rows, const size_t& cols, const T& init = T())
        : n_rows(rows), n_cols(cols), data(rows, std::vector<T>(cols, init)) {
    }

    Matrix& operator=(const Matrix& other) = default;

    Matrix& operator=(Matrix&& other) noexcept = default;

    explicit Matrix(const std::vector<std::vector<T>>& _data)
        : n_rows(_data.size()), n_cols(_data[0].size()) {

        data = std::vector<std::vector<T>>(n_rows, std::vector<T>(n_cols, T()));

        for (size_t i = 0; i < _data.size(); ++i)
            for (size_t j = 0; j < _data[0].size(); ++j)
                data[i][j] = _data[i][j];
    }

    explicit Matrix(const std::vector<T>& _data, const int& _axis = 0) {

        if (_axis == 0) {
            n_rows = _data.size();
            n_cols = 1;

            data = std::vector<std::vector<T>>(n_rows, std::vector<T>(n_cols, T()));
            for (size_t i = 0; i < _data.size(); ++i)
                data[i][0] = _data[i];
        }
        else if (_axis == 1) {
            n_rows = 1;
            n_cols = _data.size();

            data = std::vector<std::vector<T>>(n_rows, std::vector<T>(n_cols, T()));
            for (size_t i = 0; i < _data.size(); ++i)
                data[0][i] = _data[i];
        }
        else {
            throw std::invalid_argument("Eje invalido");
        }


    }

    explicit Matrix(const T* _flatten_matrix, const size_t& _size, const size_t& _rows, const size_t& _cols)
        : Matrix(_rows, _cols) {

        if (_size != _rows * _cols)
            throw std::invalid_argument("E");

        for (int row = 0, i = 0; row < _rows; ++row) {
            for (int col = 0; col < _cols; ++col, ++i)
                data[row][col] = _flatten_matrix[i];
        }
    }


    explicit Matrix(const std::vector<double>& _flatten_matrix, const size_t& _size, const size_t& _rows, const size_t& _cols)
        : Matrix(_rows, _cols) {

        if (_size != _rows * _cols)
            throw std::invalid_argument("E");

        for (int row = 0, i = 0; row < _rows; ++row) {
            for (int col = 0; col < _cols; ++col, ++i)
                data[row][col] = _flatten_matrix[i];
        }
    }


    size_t rows() const { return n_rows; }
    size_t cols() const { return n_cols; }

    std::vector<T>& operator[](size_t i) {
        if (i >= n_rows)
            throw std::out_of_range("Indice fuera de rango en operator[]");
        return data[i];
    }

    const std::vector<T>& operator[](size_t i) const {
        if (i >= n_rows)
            throw std::out_of_range("Indice fuera de rango en operator[]");
        return data[i];
    }

    Matrix apply_function(const std::function<T(T)>& function) const {
        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                result.data[i][j] = function(data[i][j]);
        return result;
    }

    Matrix transpose() const {
        Matrix result(n_cols, n_rows);
        for (size_t i = 0; i < n_cols; ++i)
            for (size_t j = 0; j < n_rows; ++j)
                result.data[i][j] = data[j][i];
        return result;
    }

    Matrix outter_product(const Matrix& _b) const {
        if (n_cols != 1 || _b.cols())
            throw std::invalid_argument("e");
        return this * _b.transpose();
    }

    std::vector<T> flat() const {
        std::vector<T> flat_matrix;
        flat_matrix.reserve(n_rows * n_cols);

        for (const auto& row : data)
            flat_matrix.insert(flat_matrix.end(), row.cbegin(), row.cend());

        return flat_matrix;
    }

    Matrix operator+(const Matrix& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols)
            throw std::invalid_argument("Dimensiones incompatibles para suma");

        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols)
            throw std::invalid_argument("Dimensiones incompatibles para resta");

        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (n_cols != other.n_rows)
            throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");

        Matrix result(n_rows, other.n_cols, T());
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < other.n_cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < n_cols; ++k)
                    sum += data[i][k] * other.data[k][j];
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    Matrix operator*(const T& scalar) const {
        Matrix result(n_rows, n_cols);
        for (size_t i = 0; i < n_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << "[";
        for (size_t i = 0; i < m.n_rows; ++i) {
            os << "[ ";
            for (size_t j = 0; j < m.n_cols; ++j) {
                os << m.data[i][j];
                if (j + 1 < m.n_cols) os << ", ";
            }
            os << " ]";
            if (i + 1 < m.n_rows) os << "\n";
        }
        os << "]";
        return os;
    }

    friend std::ifstream& operator>>(std::ifstream& in, Matrix& m) {
        in.read(reinterpret_cast<char*>(&m.n_rows), sizeof(m.n_rows));
        in.read(reinterpret_cast<char*>(&m.n_cols), sizeof(m.n_cols));

        m.data.resize(m.n_rows, std::vector<T>(m.n_cols));
        for (auto& row : m.data)
            in.read(reinterpret_cast<char*>(row.data()), m.n_cols * sizeof(T));

        return in;
    }

    friend std::ofstream& operator<<(std::ofstream& out, const Matrix& m) {
        out.write(reinterpret_cast<const char*>(&m.n_rows), sizeof(m.n_rows));
        out.write(reinterpret_cast<const char*>(&m.n_cols), sizeof(m.n_cols));

        for (const auto& row : m.data)
            out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));

        return out;
    }
};

#endif