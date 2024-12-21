# File Generator

This repository contains Python scripts for generating binary files using both CPU and GPU resources. The GPU implementation leverages CUDA for parallel processing.

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

The File Generator project provides tools to create large binary files filled with random data. This is particularly useful for testing storage systems, benchmarking, and other scenarios where large datasets are required. The repository includes two main scripts:

- `BinFileGen-CPU.py`: Generates binary files using the CPU.
- `BinFileGen-GPU.py`: Generates binary files using the GPU with CUDA support.

## Features

- Generate binary files of specified sizes.
- Option to utilize GPU acceleration for faster file generation.
- Customizable parameters for data patterns and file sizes.

## Requirements

- Python 3.x
- NumPy library
- For GPU-based generation:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit installed
  - PyCUDA library

## Usage

### CPU-Based File Generation

To generate a binary file using the CPU script:

1. Ensure Python 3.x and NumPy are installed.
2. Run the script with the desired parameters:

   ```bash
   python BinFileGen-CPU.py --output <output_file> --size <file_size>
   ```

   - `<output_file>`: Path to the output binary file.
   - `<file_size>`: Size of the file to generate (in bytes).

### GPU-Based File Generation

To generate a binary file using the GPU script:

1. Ensure the following are installed:
   - Python 3.x
   - NumPy
   - NVIDIA GPU with CUDA support
   - CUDA Toolkit
   - PyCUDA
2. Run the script with the desired parameters:

   ```bash
   python BinFileGen-GPU.py --output <output_file> --size <file_size>
   ```

   - `<output_file>`: Path to the output binary file.
   - `<file_size>`: Size of the file to generate (in bytes).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [NumPy Documentation](https://numpy.org/doc/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

```

This `README.md` provides an overview of your project, instructions for usage, and relevant references to assist users in understanding and utilizing your file generator scripts effectively. 
