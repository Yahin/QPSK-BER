import numpy as np

def fir_q15_bit_accurate(x_q15, h_q15):
    """Bit-accurate emulation of ARM CMSIS-DSP arm_fir_q15."""
    N = len(h_q15)
    L = len(x_q15)
    y_q15 = np.zeros(L, dtype=np.int16)
    # Pad input with N-1 zeros on the left for the initial delay line
    padded_x = np.concatenate([np.zeros(N - 1, dtype=np.int64), x_q15.astype(np.int64)])
    for n in range(L):
        # Window of previous N samples (reversed for convolution)
        x_window = padded_x[n : n + N][::-1]
        
        # 64-bit Accumulator
        acc = np.sum(h_q15.astype(np.int64) * x_window)
        
        # Shift by 15 bits and saturate to int16
        acc_shifted = acc >> 15
        acc_sat = np.clip(acc_shifted, -32768, 32767)
        y_q15[n] = np.int16(acc_sat)
    return y_q15

def export_c_header(h_q15, x_q15, y_q15, filename="C/test_vectors.h"):
    """Exports Python golden vectors into a C header file."""
    with open(filename, "w") as f:
        f.write("#ifndef TEST_VECTORS_H\n#define TEST_VECTORS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TAPS {len(h_q15)}\n")
        f.write(f"#define BLOCK_SIZE {len(x_q15)}\n\n")
        # 1. RRC Filter Taps
        f.write("static const int16_t rrc_taps_q15[NUM_TAPS] = {\n    ")
        f.write(", ".join(map(str, h_q15)))
        f.write("\n};\n\n")
        # 2. Test Input Samples
        f.write("static const int16_t test_input_q15[BLOCK_SIZE] = {\n    ")
        f.write(", ".join(map(str, x_q15)))
        f.write("\n};\n\n")
        # 3. Expected Output from Python Golden Model
        f.write("static const int16_t expected_output_q15[BLOCK_SIZE] = {\n    ")
        f.write(", ".join(map(str, y_q15)))
        f.write("\n};\n\n")
        f.write("#endif // TEST_VECTORS_H\n")
    print(f"Successfully exported {filename}!")


def rrc_taps(N=31, sps=4, alpha=0.35):
    """Generates an N-tap Root-Raised Cosine filter impulse response."""
    indices = np.arange(-(N // 2), (N // 2) + 1)
    t = indices / sps
    h = np.zeros(len(t))

    for i, t_val in enumerate(t):
        if abs(t_val) < 1e-8:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif abs(abs(4 * alpha * t_val) - 1.0) < 1e-8:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * t_val * (1 - alpha)) + 4 * alpha * t_val * np.cos(np.pi * t_val * (1 + alpha))
            den = np.pi * t_val * (1 - (4 * alpha * t_val) ** 2)
            h[i] = num / den

    # Energy normalization
    h = h / np.sqrt(np.sum(h ** 2))
    return h

if __name__ == "__main__":
    N=31
    sps=4
    alpha=0.35
    
    h = rrc_taps(N, sps, alpha)
    h_q15 = np.round(h * ((2**15-1))).astype(np.int16)
    print(f"Float peak value: {np.max(h):.4f}")
    print(f"Q15 peak value (center tap): {np.max(h_q15)}")
    print(f"Float energy sum(h^2): {np.sum(h**2):.4f}")

    symbols = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1], dtype=float) / np.sqrt(2)
    upsampled_float = np.zeros(len(symbols) * sps, dtype=float)
    upsampled_float[::sps] = symbols

    x_q15 = np.round(upsampled_float * ((2**15-1))).astype(np.int16)

    y_q15 = fir_q15_bit_accurate(x_q15, h_q15)

    y_float = np.convolve(upsampled_float, h, mode='full')[:len(upsampled_float)]
    y_float_scaled = np.round(y_float * ((2**15-1)))
    diff_lsb = np.abs(y_q15 - y_float_scaled)
    print(f"Max error between Float and Q15: {np.max(diff_lsb):.1f} LSBs")
    export_c_header(h_q15, x_q15, y_q15)