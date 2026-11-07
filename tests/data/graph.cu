// The program behind nsys-2024.6-graph.sqlite. See tests/test_real.py.
// loop.cu with pressure and tail captured once into a CUDA graph. Every
// iteration launches the graph inside one NVTX "step", then synchronizes.
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
    cudaMalloc(&x, n * sizeof(float));
    cudaMemset(x, 0, n * sizeof(float));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    pressure<<<n / 256, 256, 0, s>>>(x, n, 4000);
    tail<<<n / 256, 256, 0, s>>>(x, n);
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    for (int it = 0; it < 5; ++it) {
        nvtxRangePushA("step");
        cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        nvtxRangePop();
    }
    float host = 0;
    cudaMemcpy(&host, x, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", host);
    return 0;
}
