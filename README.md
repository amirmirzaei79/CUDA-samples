# CUDA Samples

This repository contains samples of CUDA programming.

* `Test Devices`: This  code outputs (prints) some information about each CUDA capable GPU in system (including GPU Name, Compute Capability, Clock Frequency and ...)
* `Vector Add`: This code generates two vectors (fills them with i & 2 * i) and adds them together using system GPU, then calculates each element using CPU to verify GPU answer.
* `Vector Elementwise Mul`: This code generates two random vectors and multiplies them elementwise using system GPU.
* `Vector Inner Product`: This code generates two random vectors and calculates their inner product using system GPU (utilizing atomic instructions).
* `Vector Inner Product Shared Mem`: This code generates two random vectors and calculates their inner product using system GPU (utilizing atomic instructions and shared memory).
* `Matrix Mul`: This code generates two random matrices, multiplies them together using system GPU, then calculates each element using CPU to verify GPU answer.

