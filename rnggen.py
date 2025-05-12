import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import sys
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import hashlib  # For additional hashing

# --- Configuration ---
NUM_BITS_NEEDED = 1024 * 1024 * 8 * 13  # Generate 1 MiB worth of bits (adjust as needed)
OUTPUT_FILENAME = "random_output.bin"
HASH_FILENAME = "random_output_hash.bin"
BRIGHTNESS_MIN = 2
BRIGHTNESS_MAX = 253
CAMERA_INDEX = 0  # 0 is usually the default built-in camera

# --- Algorithm Implementation ---

def generate_random_bits(num_bits_to_generate):
    """
    Implements Algorithm 1 to generate random bits using the camera.
    """
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        print("Please ensure a camera is connected and drivers are installed.")
        print("Try changing CAMERA_INDEX if you have multiple cameras.")
        sys.exit(1)  # Exit if camera cannot be opened

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    # Variables
    final_list_bits = []
    num_so_far = 0
    frame_count = 0

    print(f"Starting generation of {num_bits_to_generate} bits...")
    start_time = time.time()

    # --- Generation Loop ---
    while num_so_far < num_bits_to_generate:
        # C1: Take one snapshot
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to capture frame. Skipping.")
            time.sleep(0.1)  # Avoid busy-waiting if camera fails temporarily
            continue

        frame_count += 1

        # Convert to grayscale to get brightness values
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pick out the brightness values within the specified range [2, 253]
        valid_pixels = gray_frame[(gray_frame >= BRIGHTNESS_MIN) & (gray_frame <= BRIGHTNESS_MAX)]

        if valid_pixels.size == 0:
            continue  # Skip if no valid pixels found

        # Take the last bits (LSB - Least Significant Bit) as a SubList
        sub_list_bits_np = valid_pixels & 1  # Use bitwise AND to get the LSB

        # If (Frame is even) flip the bits in SubList
        if frame_count % 2 == 0:
            sub_list_bits_np = 1 - sub_list_bits_np  # Flip 0s to 1s and 1s to 0s

        # Convert numpy array to list for extending
        sub_list = sub_list_bits_np.tolist()

        # C2: Add SubList to FinalList (row-major is implicit when extending a list)
        final_list_bits.extend(sub_list)

        # C3: Update count
        num_so_far += len(sub_list)

        # Optional: Display progress
        if frame_count % 30 == 0:  # Update every ~second assuming ~30fps
            progress = min(100, (num_so_far / num_bits_to_generate) * 100)
            print(f"Progress: {progress:.2f}% ({num_so_far} / {num_bits_to_generate} bits)", end='\r')

    # --- End While ---
    end_time = time.time()
    print(f"\nGeneration complete. Collected {num_so_far} bits.")

    # --- Timing End ---
    duration = end_time - start_time
    bits_per_second = num_so_far / duration if duration > 0 else 0
    print(f"Total time: {duration:.2f} seconds")
    print(f"Generation speed: {bits_per_second:.2f} bits/second ({bits_per_second/8/1024:.2f} KB/s)")

    # Release camera
    cap.release()
    cv2.destroyAllWindows()

    return final_list_bits

def save_bits_to_binary_file(bits, filename):
    """
    Packs a list of bits into bytes and saves to a binary file.
    """
    print(f"Packing bits and saving to '{filename}'...")
    remainder = len(bits) % 8
    if remainder != 0:
        padding = 8 - remainder
        bits.extend([0] * padding)
        print(f"Note: Padded with {padding} zero bits to align to bytes.")

    bit_array = np.array(bits, dtype=np.uint8)
    byte_array = np.packbits(bit_array)

    try:
        with open(filename, 'wb') as f:
            f.write(byte_array.tobytes())
        print(f"Successfully saved {len(byte_array)} bytes to {filename}")
    except IOError as e:
        print(f"Error: Could not write to file {filename}. {e}")
        sys.exit(1)

    # Remove 0 and 255 values from the byte array
    byte_array = byte_array[(byte_array != 0) & (byte_array != 255)]
    return byte_array

def hash_with_aes(byte_data, hash_filename):
    """
    Hashes the byte data using AES in CBC mode and saves the hash to a file.
    """
    print("Hashing data with AES...")
    key = get_random_bytes(16)  # Generate a random 16-byte key
    cipher = AES.new(key, AES.MODE_CBC)  # Create AES cipher in CBC mode
    iv = cipher.iv  # Initialization vector

    # Convert numpy array to bytes
    byte_data = byte_data.tobytes()

    # Pad the data to make it a multiple of the AES block size (16 bytes)
    padded_data = pad(byte_data, AES.block_size)

    # Encrypt the data
    encrypted_data = cipher.encrypt(padded_data)

    # Save the hash (encrypted data) to a file
    try:
        with open(hash_filename, 'wb') as f:
            f.write(iv + encrypted_data)  # Save IV + encrypted data
        print(f"Successfully saved AES hash to {hash_filename}")
    except IOError as e:
        print(f"Error: Could not write to file {hash_filename}. {e}")
        sys.exit(1)

    return encrypted_data

def plot_histogram(byte_data, title="Histogram of Generated Byte Values"):
    """
    Plots a histogram of the byte values (0-255).
    """
    print("Generating histogram...")
    if byte_data is None or len(byte_data) == 0:
        print("No data to plot histogram.")
        return

    # Ensure byte_data is a numpy array of uint8
    if not isinstance(byte_data, np.ndarray):
        byte_data = np.frombuffer(byte_data, dtype=np.uint8)

    plt.figure(figsize=(10, 6))
    counts, bin_edges, patches = plt.hist(byte_data, bins=256, range=(0, 256), density=False, alpha=0.75)
    plt.xlabel("Byte Value (0-255)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', alpha=0.5)

    expected_count = len(byte_data) / 256
    plt.axhline(expected_count, color='r', linestyle='dashed', linewidth=1, label=f'Expected Uniform Count ({expected_count:.2f})')
    plt.legend()

    print("Displaying histogram...")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    generated_bits = generate_random_bits(NUM_BITS_NEEDED)
    generated_bytes = save_bits_to_binary_file(generated_bits, OUTPUT_FILENAME)
    encrypted_data = hash_with_aes(generated_bytes, HASH_FILENAME)
    plot_histogram(generated_bytes, title="Histogram of Generated Byte Values")
    plot_histogram(encrypted_data, title="Histogram of Encrypted Data Values")
    print("Done.")