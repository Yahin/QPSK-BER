import numpy as np

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_levels = 1 - 2 * i_bits
    q_levels = 1 - 2 * q_bits

    s = (i_levels + 1j* q_levels)/np.sqrt(2)

    return s

def add_noise(symbols: np.ndarray, eb_no_db: float) -> np.ndarray:
    eb_no_linear = 10**(eb_no_db/10)
    Es = 1
    Eb = Es / 2
    sigma = np.sqrt(Eb/(2*eb_no_linear))


if __name__ == "__main__":
    test_bits = np.array([0, 0,  0, 1,  1, 0,  1, 1])
    symbols = qpsk_modulate(test_bits)
    print("Modulated Symbols:\n", symbols)
    print("Average Symbol Power:", np.mean(np.abs(symbols)**2))
