# BinFileGen-GPU.py
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

def generate_binary_file(file_name, file_size):
    start_time = time.time()

    # Initialize xoroshiro128plus state
    state = np.array([np.uint64(0xdeadbeef), np.uint64(0xbeefdead)], dtype=np.uint64)
    state_gpu = cuda.mem_alloc(state.nbytes)
    cuda.memcpy_htod(state_gpu, state)

    # Prepare output array on GPU
    output = np.zeros(file_size, dtype=np.uint8)
    output_gpu = cuda.mem_alloc(output.nbytes)

    # Generate random bytes on GPU
    block_size = 256
    grid_size = (file_size + block_size - 1) // block_size
    generate_random_bytes(output_gpu, state_gpu, np.int64(file_size), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy data back to host
    cuda.memcpy_dtoh(output, output_gpu)

    # Write to binary file
    with open(file_name, 'wb') as f:
        f.write(output)

    time_taken = time.time() - start_time
    print(f"Binary file '{file_name}' generated with size {file_size} bytes in {time_taken}s")

def parse_size(size_str):
    size_str = size_str.strip().upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)  

# CUDA kernel code
kernel_code = """
extern "C" {
    __device__ unsigned long long xoroshiro128plus(unsigned long long *s)
    {
        unsigned long long s0 = s[0];
        unsigned long long s1 = s[1];
        unsigned long long result = s0 + s1;

        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14); // a, b
        s[1] = (s1 << 36) | (s1 >> 28); // c

        return result;
    }

    __global__ void generate_random_bytes(unsigned char *output, unsigned long long *state, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            unsigned long long random_value = xoroshiro128plus(state);
            output[idx] = random_value & 0xFF;
        }
    }
}
"""

# Compile the CUDA kernel
mod = SourceModule(kernel_code)

# Get the function from the compiled module
generate_random_bytes = mod.get_function("generate_random_bytes")


if __name__ == "__main__":
    
    # Input: File Name
    file_name = input("Enter the file name: ") + '.bin'

    # Input: File Size
    while True:
        try:
            raw_file_size = input("Enter size (e.g., '100KB', '2MB', '1GB', or raw integer for B): ")
            file_size = parse_size(raw_file_size)
            break
        except :
            print("Invalid size format. Please use a format like '100KB', '2MB', '1GB', or a raw integer for B")
   
    # File Generation
    generate_binary_file(file_name, file_size)
