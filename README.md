# File Generator

This repository contains Python scripts for generating binary files using either the CPU or GPU. The GPU implementation uses PyCUDA for parallel processing, while the CPU implementation uses Python's `random` module.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [CPU-Based File Generation](#cpu-based-file-generation)
  - [GPU-Based File Generation](#gpu-based-file-generation)
- [License](#license)
- [References](#references)

## Introduction

The File Generator project offers tools to create large binary files filled with random data. It includes two scripts:

- `BinFileGen-CPU.py`: A script that generates binary files using the CPU and Python's `random` module.
- `BinFileGen-GPU.py`: A script that utilizes PyCUDA and the `xoroshiro128plus` algorithm to generate binary files on the GPU.

## Features

- Generate binary files of custom sizes.
- Flexible input formats for file size (e.g., `100KB`, `2MB`, `1GB`).
- CPU and GPU-based implementations for file generation.
- Parallelized random byte generation on GPU for improved performance.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pycuda`
- For GPU-based generation:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit installed

## Usage

### CPU-Based File Generation

The CPU-based script (`BinFileGen-CPU.py`) generates binary files using the `random` module.

#### Steps to Run:

1. Ensure Python 3.x is installed.
2. Install the required libraries:
   ```bash
   pip install numpy
   ```
3. Run the script:
   ```bash
   python BinFileGen-CPU.py
   ```
4. Provide the following inputs:
   - File name (without extension).
   - File size (e.g., `100KB`, `2MB`, `1GB`, or a raw integer for bytes).

Example:
```bash
Enter the file name: test_file
Enter size (e.g., '100KB', '2MB', '1GB', or raw integer for B): 500MB
```

Output:
```bash
Binary file 'test_file.bin' generated with size 524288000 bytes in X.XXs
```

### GPU-Based File Generation

The GPU-based script (`BinFileGen-GPU.py`) generates binary files using CUDA and the `xoroshiro128plus` random number generation algorithm.

#### Steps to Run:

1. Ensure you have an NVIDIA GPU with CUDA support.
2. Install the required libraries:
   ```bash
   pip install numpy pycuda
   ```
3. Ensure the CUDA kernel file (`xoroshiro128plus_kernel.cu`) is in the same directory as the script.
4. Run the script:
   ```bash
   python BinFileGen-GPU.py
   ```
5. Provide the following inputs:
   - File name (without extension).
   - File size (e.g., `100KB`, `2MB`, `1GB`, or a raw integer for bytes).

Example:
```bash
Enter the file name: test_file
Enter size (e.g., '100KB', '2MB', '1GB', or raw integer for B): 1GB
```

Output:
```bash
Binary file 'test_file.bin' generated with size 1073741824 bytes in X.XXs
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [NumPy Documentation](https://numpy.org/doc/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
