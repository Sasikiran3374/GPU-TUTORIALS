Tensor Core Matrix Multiplication

Multiplies two 64×64 matrices using NVIDIA Tensor Cores with 16×16 WMMA tiles.
The computation uses 4×4 = 16 output tiles, with each tile accumulated across the 4 K-dimension tiles.
The CUDA program verifies the Tensor Core result against a CPU matrix multiplication and checks that both results match.
Result: Tensor Core Matrix Multiplication: SUCCESS
