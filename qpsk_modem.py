import numpy as np

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Modulates a binary bitstream into Gray-coded QPSK complex symbols (Es = 1.0).
    Bit-to-Quadrant Gray Mapping:
      (b0, b1) = (I_bit, Q_bit) where bit 0 -> +1/sqrt(2), bit 1 -> -1/sqrt(2)
      
      Quadrant 1 (b0=0, b1=0): (+1 + 1j) / sqrt(2)
      Quadrant 2 (b0=1, b1=0): (-1 + 1j) / sqrt(2)
      Quadrant 3 (b0=1, b1=1): (-1 - 1j) / sqrt(2)
      Quadrant 4 (b0=0, b1=1): (+1 - 1j) / sqrt(2)
      
    Adjacent quadrants differ by exactly 1 bit (Gray coded), minimizing bit errors
    when nearest-neighbor symbol decision errors occur.
    """
    if len(bits) % 2 != 0:
        raise ValueError("Input bits array length must be even for QPSK.")
    
    bits = bits.astype(float)
    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_levels = 1.0 - 2.0 * i_bits
    q_levels = 1.0 - 2.0 * q_bits

    s = (i_levels + 1j* q_levels)/np.sqrt(2) 
    
    return s

def add_noise(symbols: np.ndarray, eb_no_db: float= 10) -> np.ndarray:
    eb_no_linear = 10**(eb_no_db/10)
    Es = 1.0
    Eb = Es / 2.0
    sigma = np.sqrt(Eb/(2*eb_no_linear))
    noise_i =  np.random.normal(0, sigma, len(symbols))
    noise_q =  np.random.normal(0, sigma, len(symbols))
    noise = noise_i + 1j* noise_q
    signal = symbols + noise
    
    return signal

def qpsk_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    """
    Demodulates noisy QPSK symbols using Maximum Likelihood (ML) sign slicing.
    Decision Boundary:
      real >= 0 -> I_bit = 0,  real < 0 -> I_bit = 1
      imag >= 0 -> Q_bit = 0,  imag < 0 -> Q_bit = 1
    Reconstitutes the serial bitstream: [I0, Q0, I1, Q1, ...].
    """
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
    tx_bits = np.random.randint(0, 2, 1_000_000)
    symbols = qpsk_modulate(tx_bits)
    noisy_symbols = add_noise(symbols,10)
    rx_bits = qpsk_demodulate(noisy_symbols)
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    
    print("Modulated Symbols:\n", symbols)
    print("Average Symbol Power:", np.mean(np.abs(symbols)**2))
    
    print("Demodulated Bits:\n", rx_bits)
    print(f"BER:{ber}")
