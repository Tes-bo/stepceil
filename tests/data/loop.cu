// The program behind nsys-2024.6-loop.sqlite. See tests/test_real.py.
// A training-loop shape: one NVTX "step" per iteration.
// Iteration 0 allocates. Every iteration launches pressure, then a small
// kernel, then synchronizes.
#include <cstdio>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

__global__ void pressure(float *x, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    for (int r = 0; r < rounds; ++r) v = v * 1.0001f + 0.5f;
    x[i] = v;
}

__global__ void tail(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 0.5f;
}

int main() {
    const int n = 1 << 20;
    float *x = nullptr;
    float *scratch = nullptr;
    for (int it = 0; it < 5; ++it) {
        nvtxRangePushA("step");
        if (it == 0) {
            cudaMalloc(&x, n * sizeof(float));
            cudaMalloc(&scratch, 256 << 20);
            cudaMemset(x, 0, n * sizeof(float));
        }
        nvtxRangePushA("forward");
        pressure<<<n / 256, 256>>>(x, n, 4000);
        tail<<<n / 256, 256>>>(x, n);
        nvtxRangePop();
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
    float host = 0;
    cudaMemcpy(&host, x, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", host);
    cudaFree(scratch);
    cudaFree(x);
    return 0;
}
