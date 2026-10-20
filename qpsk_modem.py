import numpy as np

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_levels = 1 - 2 * i_bits
    q_levels = 1 - 2 * q_bits

    s = (i_levels + 1j* q_levels)/np.sqrt(2) 
    symbols = add_noise(s, 0)
    
    return symbols

def add_noise(symbols: np.ndarray, eb_no_db: float) -> np.ndarray:
    eb_no_linear = 10**(eb_no_db/10)
    Es = 1
    Eb = Es / 2
    sigma = np.sqrt(Eb/(2*eb_no_linear))
    noise_i =  np.random.normal(0, sigma, len(symbols))
    noise_q =  np.random.normal(0, sigma, len(symbols))
    noise = noise_i + 1j* noise_q
    signal = symbols + noise
    
    return signal

def qpsk_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    real_part = np.real(received_symbols)
    imag_part = np.imag(received_symbols)

    i_bits = (real_part < 0).astype(int)
    q_bits = (imag_part < 0).astype(int)
    
    bits = np.empty(len(received_symbols)*2, dtype=int)
    bits[0::2] = i_bits
    bits[1::2] = q_bits

    return bits


if __name__ == "__main__":
    #tx_bits = np.array([0, 1,  0, 1,  1, 0,  1, 1])
    tx_bits = np.random.randint(0, 2, 100_000)
    symbols = qpsk_modulate(tx_bits)
    rx_bits = qpsk_demodulate(symbols)
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    
    print("Modulated Symbols:\n", symbols)
    print("Average Symbol Power:", np.mean(np.abs(symbols)**2))
    
    print("Demodulated Bits:\n", rx_bits)
    print(f"BER:{ber}")
