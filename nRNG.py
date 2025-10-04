import numpy as np
import hashlib

# --- Von Neumann extractor ---
def von_neumann_extraction(folded_bits):
    extracted_bits = []
    for i in range(0, len(folded_bits) - 1, 2):
        pair = folded_bits[i], folded_bits[i + 1]
        if pair == (0,1) or pair == (1,0):
            extracted_bits.append(folded_bits[i])
    return np.array(extracted_bits)

# --- Convert bits to bytes ---
def bits_to_bytes(bits): 
    pad_len = (8 - len(bits) % 8) % 8
    bits_padded = np.concatenate((bits, np.zeros(pad_len, dtype=int)))
    return np.packbits(bits_padded).tobytes()

# --- SHA256 extractor ---
def sha256_extractor(bits):
    block_size = 256
    final_bits = []
    for i in range(0, len(bits), block_size):
        block = bits[i:i + block_size]
        if len(block) < block_size:
            break
        hash_bytes = hashlib.sha256(bits_to_bytes(block)).digest()
        hash_bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        final_bits.extend(hash_bits[:block_size])
    return np.array(final_bits)

# --- Generate raw bits from noise ---
def generate_bits():
    dt = 0.001
    t = np.arange(0, 15000, dt) 
    f = 2.5 * np.random.randn(len(t))
    n = len(t)
    fhat = np.fft.fft(f, n)
    PSD = fhat * np.conj(fhat) / n

    threshold = np.median(PSD)
    bits = (PSD > threshold).astype(int)
    half = len(bits) // 2
    folded_bits = bits[:half] ^ bits[half:half*2]

    return von_neumann_extraction(folded_bits)

# --- Generate single digit (0-9) ---
def generate_single_digit():
    raw_bits = generate_bits()
    final_bits = sha256_extractor(raw_bits)

    if len(final_bits) < 4:
        raise ValueError("Not enough bits generated.")

    rand_int = int(''.join(map(str, final_bits[:4])), 2)
    single_digit = rand_int % 10
    return single_digit

# --- Generate multiple digits ---
def generate_multiple_digits(N):
    digits = [generate_single_digit() for _ in range(N)]
    return np.array(digits)

if __name__ == "__main__":
    N = 8
    phi = generate_multiple_digits(N)
    print("Random φ:", phi)
