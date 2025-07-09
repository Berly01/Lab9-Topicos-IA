# Lab9-Topicos-IA

## Backward Propagation Completo

### Gradientes de Kernels: Calculados mediante convolución entre entrada y gradiente de salida
### Gradientes de Entrada: Calculados mediante convolución transpuesta con kernels volteados
### Gradientes de Activación: Derivadas de ReLU, Sigmoid y Tanh
### Gradientes de Pooling:

. Max/Min pooling: gradiente se propaga solo al elemento seleccionado
. Average pooling: gradiente se distribuye uniformemente


```cpp
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

```

### Estructuras de Cache

. ForwardCache: Almacena salidas intermedias necesarias para el backward pass
. Índices de Pooling: Guarda las posiciones de los elementos seleccionados en max/min pooling

### Capas Densas (DenseLayer)

. Implementación completa con forward y backward pass
