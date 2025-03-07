import os
import string
import random

def generate_text_file(file_name, size_in_mb):
    """
    Generate a text file of a specific size.
    
    Parameters:
    file_name (str): Name of the file to generate.
    size_in_mb (float): Size of the file in megabytes (MB).
    """
    size_in_bytes = int(size_in_mb * 1024 * 1024)  # Convert MB to bytes
    chunk_size = 16 * 1024 * 1024  # Write in 16 MB chunks
    characters = string.ascii_letters + string.digits + " \n"  # Random content
    
    try:
        with open(file_name, 'w') as file:
            bytes_written = 0
            while bytes_written < size_in_bytes:
                # Generate a chunk of random text
                remaining_bytes = size_in_bytes - bytes_written
                current_chunk_size = min(chunk_size, remaining_bytes)
                text_chunk = ''.join(random.choices(characters, k=current_chunk_size))
                file.write(text_chunk)
                bytes_written += current_chunk_size
        
        print(f"File '{file_name}' of size {size_in_mb} MB created successfully.")
    except Exception as e:
        print(f"Error generating file: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a text file of a specific size.")
    parser.add_argument("file_name", help="Name of the file to generate")
    parser.add_argument("size", type=float, help="Size of the file in MB")
    args = parser.parse_args()

    generate_text_file(args.file_name, args.size)
