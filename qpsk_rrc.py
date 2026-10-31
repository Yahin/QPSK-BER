import numpy as np
import matplotlib.pyplot as plt
from qpsk_modem import qpsk_modulate, qpsk_demodulate, add_noise
from design_rrc import rrc_taps

# Parameters
sps = 4           # Samples per symbol
N = 31            # RRC Filter order (taps)
alpha = 0.35      # Roll-off factor
num_bits = 1000   # 1000 bits = 500 QPSK symbols

# 1. Generate RRC filter taps
h = rrc_taps(N=N, sps=sps, alpha=alpha)

# 2. Generate random bits and modulate to QPSK symbols
tx_bits = np.random.randint(0, 2, num_bits)
symbols = qpsk_modulate(tx_bits)
upsampled = np.zeros(len(symbols)*sps, dtype=complex)
upsampled[::4] = symbols

tx_waveform = np.convolve(upsampled, h, mode='full')
tx_noisy = add_noise(tx_waveform,10)

rx_matched = np.convolve(tx_noisy, h, mode='full')
sample_indices = 30 + np.arange(len(symbols)) * sps
recovered_symbols = rx_matched[sample_indices]

rx_bits = qpsk_demodulate(recovered_symbols)
errors = np.sum(tx_bits != rx_bits)
print(f"Total Bit Errors: {errors} / {len(tx_bits)}")
