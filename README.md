# Lab9-Topicos-IA

## Backward Propagation Completo

### Gradientes de Kernels: Calculados mediante convolución entre entrada y gradiente de salida
### Gradientes de Entrada: Calculados mediante convolución transpuesta con kernels volteados
### Gradientes de Activación: Derivadas de ReLU, Sigmoid y Tanh
### Gradientes de Pooling:

. Max/Min pooling: gradiente se propaga solo al elemento seleccionado
. Average pooling: gradiente se distribuye uniformemente



### Estructuras de Cache

. ForwardCache: Almacena salidas intermedias necesarias para el backward pass
. Índices de Pooling: Guarda las posiciones de los elementos seleccionados en max/min pooling

### Capas Densas (DenseLayer)

. Implementación completa con forward y backward pass
