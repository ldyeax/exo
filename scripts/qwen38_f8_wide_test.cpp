#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float f8_to_f32(uint8_t value) {
    const float sign = (value & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;
    if (exponent == 0) {
        return sign * std::ldexp(float(mantissa), -9);
    }
    return sign * std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7);
}

static bool run_case(ggml_backend_t backend, int ncols) {
    constexpr int k = 256;
    constexpr int n = 64;

    ggml_init_params params = {};
    params.mem_size = 8 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(context, GGML_TYPE_F8_E4M3, k, n);
    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, k, ncols);
    ggml_tensor * output = ggml_mul_mat(context, weights, input);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        ggml_free(context);
        return false;
    }

    constexpr int block_bytes = 130;
    constexpr int blocks_per_row = k / 128;
    constexpr std::array<uint16_t, 4> scales_bf16 = {
        0x3f80, // 1.0
        0x3f00, // 0.5
        0x3e80, // 0.25
        0x4000, // 2.0
    };
    constexpr std::array<float, 4> scales = {1.0f, 0.5f, 0.25f, 2.0f};

    std::vector<uint8_t> packed(n * blocks_per_row * block_bytes);
    std::vector<uint8_t> codes(n * k);
    std::vector<float> weight_values(n * k);
    for (int row = 0; row < n; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const int scale_index = (row + block) % int(scales.size());
            const int packed_offset = (row * blocks_per_row + block) * block_bytes;
            packed[packed_offset + 0] = uint8_t(scales_bf16[scale_index] & 0xffu);
            packed[packed_offset + 1] = uint8_t(scales_bf16[scale_index] >> 8);
            for (int offset = 0; offset < 128; ++offset) {
                const int col = block * 128 + offset;
                uint8_t code = uint8_t((row * 29 + col * 7) % 0x77);
                if ((row + col) & 1) {
                    code |= 0x80u;
                }
                packed[packed_offset + 2 + offset] = code;
                codes[row * k + col] = code;
                weight_values[row * k + col] = f8_to_f32(code) * scales[scale_index];
            }
        }
    }

    std::vector<float> host_input(k * ncols);
    std::vector<_Float16> half_input(k * ncols);
    for (int col = 0; col < ncols; ++col) {
        for (int index = 0; index < k; ++index) {
            const float value = std::sin(float(index * 3 + col * 11) * 0.017f) *
                (0.125f + 0.0005f * float(col));
            host_input[col * k + index] = value;
            half_input[col * k + index] = (_Float16) value;
        }
    }

    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, host_input.data(), 0, host_input.size() * sizeof(float));

    setenv("GGML_CUDA_F8_WIDE", "1", 1);
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "wide graph failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    std::vector<float> wide(n * ncols);
    ggml_backend_tensor_get(output, wide.data(), 0, wide.size() * sizeof(float));

    setenv("GGML_CUDA_F8_WIDE", "0", 1);
    status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS control graph failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    std::vector<float> control(n * ncols);
    ggml_backend_tensor_get(output, control.data(), 0, control.size() * sizeof(float));

    float max_reference = 0.0f;
    float max_wide_control_error = 0.0f;
    float max_wide_reference_error = 0.0f;
    float max_control_reference_error = 0.0f;
    for (int col = 0; col < ncols; ++col) {
        for (int row = 0; row < n; ++row) {
            float reference = 0.0f;
            for (int index = 0; index < k; ++index) {
                const _Float16 half_weight = (_Float16) weight_values[row * k + index];
                reference += float(half_weight) * float(half_input[col * k + index]);
            }
            const int output_index = col * n + row;
            max_reference = std::max(max_reference, std::fabs(reference));
            max_wide_control_error = std::max(
                max_wide_control_error, std::fabs(wide[output_index] - control[output_index]));
            max_wide_reference_error = std::max(
                max_wide_reference_error, std::fabs(wide[output_index] - reference));
            max_control_reference_error = std::max(
                max_control_reference_error, std::fabs(control[output_index] - reference));
        }
    }

    const float relative_wide_control = max_wide_control_error / max_reference;
    std::printf(
        "ncols=%d max_ref=%g wide_vs_cublas=%g rel=%g wide_vs_ref=%g cublas_vs_ref=%g\n",
        ncols,
        max_reference,
        max_wide_control_error,
        relative_wide_control,
        max_wide_reference_error,
        max_control_reference_error);

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return std::isfinite(relative_wide_control) && relative_wide_control < 0.005f;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "CUDA backend init failed\n");
        return 2;
    }

    bool passed = true;
    for (const int ncols : {16, 32, 64, 128, 256, 512}) {
        passed = run_case(backend, ncols) && passed;
    }
    ggml_backend_free(backend);
    std::puts(passed ? "PASS: F8_E4M3 tiled W8A16 widths 16 through 512" : "FAIL");
    return passed ? 0 : 1;
}
