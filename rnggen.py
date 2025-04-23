
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import math
from tqdm import tqdm
import sys
import time

def setup_camera(index=0, width=640, height=480):
    """Initializes the camera."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć kamery o indeksie {index}.", file=sys.stderr)
        return None
    # Optional: Set resolution (might not be supported by all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Kamera {index} otwarta.")
    # Allow camera to warm up/adjust
    time.sleep(1)
    return cap

def capture_frame(cap):
    """Captures a single frame from the camera."""
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Błąd: Nie można odczytać klatki z kamery.", file=sys.stderr)
        return None
    return frame # Frame is typically in BGR format

def process_frame(frame, frame_index):
    """
    Processes a single frame according to the algorithm:
    1. Filter pixels based on brightness [2, 253] for each RGB channel.
    2. Extract the Least Significant Bit (LSB) from valid channels.
    3. Flip bits if the frame index is odd (matching diagram: Frame 2=idx 1 flips, Frame 4=idx 3 flips).
    Returns a list of bits.
    """
    if frame is None:
        return []

    # Create a mask for pixels/channels within the brightness range [2, 253]
    # Apply to each channel independently
    mask = (frame >= 2) & (frame <= 253) # Shape: (height, width, 3)

    # Extract the values that satisfy the mask
    valid_channel_values = frame[mask] # Flattens into 1D array

    # Extract the LSB (value & 1)
    lsb_bits = valid_channel_values & 1

    # --- Bit Flipping ---
    # The pseudocode says "If (Frame is even) flip".
    # The diagram shows Frame 2 (index 1) and Frame 4 (index 3) flipping.
    # We follow the DIAGRAM's logic: flip on ODD indices (1, 3, 5...).
    if frame_index % 2 != 0: # Flip bits for odd frame indices (1, 3, ...)
        sub_list = lsb_bits[::-1].tolist() # Reverse the order of bits]
    else:
        sub_list = lsb_bits.tolist() # Keep bits as they are

    return sub_list

def bits_to_bytes(bits):
    """Converts a list of bits (0s and 1s) into a byte string."""
    if not bits:
        return b''

    # Pad with 0s at the end to make the total length a multiple of 8
    # This aligns with pseudocode C4 "Extra bits are appended..." interpreted
    # as needing byte-alignment for file output.
    padding = (8 - len(bits) % 8) % 8
    padded_bits = bits + [0] * padding

    byte_array = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | padded_bits[i+j]
        byte_array.append(byte_val)

    return bytes(byte_array)

def main():
    parser = argparse.ArgumentParser(description="Generuje losowe bity używając kamery internetowej zgodnie z Algorytmem 1.")
    parser.add_argument("-n", "--num-bits", type=int, default=13 * 1024 * 1024 * 8,
                        help="Liczba bitów do wygenerowania (domyślnie: 13MB ≈ 109 milionów bitów)")
    parser.add_argument("-o", "--output-file", type=str, default="random_bits.bin",
                        help="Plik binarny do zapisu wygenerowanych bitów (domyślnie: random_bits.bin)")
    parser.add_argument("-c", "--camera-index", type=int, default=0,
                        help="Indeks kamery do użycia (domyślnie: 0)")
    parser.add_argument("-w", "--width", type=int, default=640, help="Szerokość klatki kamery (domyślnie: 640)")
    parser.add_argument("-H", "--height", type=int, default=480, help="Wysokość klatki kamery (domyślnie: 480)")

    args = parser.parse_args()

    num_needed_bits = args.num_bits
    output_filename = args.output_file
    camera_index = args.camera_index
    frame_width = args.width
    frame_height = args.height

    print(f"Żądana liczba bitów: {num_needed_bits}")
    print(f"Plik wyjściowy: {output_filename}")
    print(f"Indeks kamery: {camera_index}")

    cap = setup_camera(camera_index, frame_width, frame_height)
    if cap is None:
        sys.exit(1) # Error message already printed by setup_camera

    final_bits = []
    num_so_far = 0
    frame_count = 0 # Start counting frames from 0

    # --- Generation Loop ---
    print("Rozpoczynanie generowania bitów...")
    try:
        # Initialize tqdm progress bar
        pbar = tqdm(total=num_needed_bits, unit='bit', desc="Generowanie", ncols=100)

        while num_so_far < num_needed_bits:
            frame = capture_frame(cap)
            if frame is None:
                print("\nBłąd odczytu klatki, przerywanie.", file=sys.stderr)
                break # Exit loop if frame capture fails

            # Process the frame
            sub_list = process_frame(frame, frame_count)

            if sub_list: # Only append and update count if bits were generated
                final_bits.extend(sub_list)
                new_bits_count = len(sub_list)
                # Update progress bar safely
                update_amount = min(new_bits_count, num_needed_bits - num_so_far)
                pbar.update(update_amount)
                num_so_far += new_bits_count

            frame_count += 1

        pbar.close() # Ensure progress bar finishes cleanly

        if num_so_far < num_needed_bits:
             print(f"\nOstrzeżenie: Wygenerowano tylko {num_so_far} z {num_needed_bits} żądanych bitów.", file=sys.stderr)
        else:
            print(f"\nZakończono generowanie. Całkowita liczba wygenerowanych bitów: {num_so_far}")
            # Note: We keep all generated bits, including extras beyond num_needed_bits,
            # as per step C4 ("Extra bits are appended").

    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.", file=sys.stderr)
        # Decide if you want to save the partially generated bits
        # For now, we'll just exit, but saving could be added here.
    finally:
        # --- Cleanup ---
        if cap and cap.isOpened():
            cap.release()
            print("Kamera zwolniona.")
        # cv2.destroyAllWindows() # Not needed as we don't show windows

    # --- Output ---
    if not final_bits:
        print("Nie wygenerowano żadnych bitów. Plik wyjściowy nie zostanie utworzony.", file=sys.stderr)
        sys.exit(1)

    print(f"Konwertowanie {len(final_bits)} bitów na bajty...")
    byte_data = bits_to_bytes(final_bits)
    num_bytes = len(byte_data)
    print(f"Rozmiar danych binarnych: {num_bytes} bajtów.")

    try:
        print(f"Zapisywanie do pliku: {output_filename}")
        with open(output_filename, 'wb') as f:
            f.write(byte_data)
        print(f"Pomyślnie zapisano {len(final_bits)} bitów ({num_bytes} bajtów) do {output_filename}")
    except IOError as e:
        print(f"Błąd: Nie można zapisać do pliku {output_filename}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()