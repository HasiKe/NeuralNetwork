#pragma once

#include <string>
#include <vector>

#include "config.hpp"

/// Compute device selection.
///
/// GPU backends are compiled in with CMake flags (NN_USE_OPENCL, NN_USE_VULKAN,
/// NN_USE_CUDA, NN_USE_HIP, NN_USE_SYCL) and selected at runtime, either in code
/// via set_backend() or with the environment variable NN_BACKEND
/// (cpu | opencl | vulkan | cuda | hip | sycl, optional NN_DEVICE=<index> and
/// NN_OFFLOAD_THRESHOLD=<flops>).
///
/// The GPU executes the matrix multiplications, which dominate the cost of
/// Dense, recurrent, convolutional and RBF layers. Small products stay on the
/// CPU, where launch and transfer overhead would outweigh the gain (see
/// set_offload_threshold()). Results are identical up to float rounding.
namespace nn {

enum class Backend : std::uint8_t { CPU = 0, OpenCL, Vulkan, CUDA, HIP, SYCL };

const char* to_string(Backend b);
/// Parse "cpu", "opencl", "vulkan", "cuda", "hip", "sycl" (case-insensitive).
Backend backend_from_string(const std::string& name);

/// Backends compiled into this build (CPU is always present).
std::vector<Backend> compiled_backends();

/// Activate a backend on the given device index. Returns false (and keeps the
/// current backend) if it is not compiled in or no usable device exists.
bool set_backend(Backend b, int device = 0);
Backend current_backend();
/// Human readable name of the active device.
std::string device_name();

/// Minimum M*N*K of a matrix product before it is sent to the GPU.
void set_offload_threshold(std::size_t flops);
std::size_t offload_threshold();

}  // namespace nn
