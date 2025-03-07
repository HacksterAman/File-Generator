# BinFileGen-CPU.py
import os
import random
import time
import concurrent.futures

def generate_chunk(chunk_size):
    return bytearray(random.getrandbits(8) for _ in range(chunk_size))

def generate_binary_file(file_name, file_size, num_workers=None):
    start_time = time.time()

    chunk_size = file_size // num_workers
    remaining_size = file_size % num_workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunks = list(executor.map(generate_chunk, [chunk_size] * num_workers))

    if remaining_size:
        chunks.append(generate_chunk(remaining_size))

    with open(file_name, 'wb') as file:
        for chunk in chunks:
            file.write(chunk)

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

if __name__ == "__main__":
    # Input: File Name
    file_name = input("Enter the file name: ") + '.bin'

    # Input: File Size
    while True:
        try:
            raw_file_size = input("Enter size (e.g., '100KB', '2MB', '1GB', or raw integer for B): ")
            file_size = parse_size(raw_file_size)
            break
        except ValueError:
            print("Invalid size format. Please use a format like '100KB', '2MB', '1GB', or a raw integer for B")

    # Determine the number of workers
    num_workers = os.cpu_count()
    
    # File Generation
    generate_binary_file(file_name, file_size, num_workers)
