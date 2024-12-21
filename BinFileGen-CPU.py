# BinFileGen-CPU.py

import os
import random
import time

def generate_binary_file(file_name, file_size):
    start_time = time.time()

    # Generate random data
    random_data = bytearray(random.getrandbits(8) for _ in range(file_size))

    # Write random data to the binary file
    with open(file_name, 'wb') as file:
        file.write(random_data)

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
        except :
            print("Invalid size format. Please use a format like '100KB', '2MB', '1GB', or a raw integer for B")
   
    # File Generation
    generate_binary_file(file_name, file_size)
