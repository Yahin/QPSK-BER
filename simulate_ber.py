import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from qpsk_modem import qpsk_modulate,add_noise, qpsk_demodulate

eb_no_dbs = np.arange(0, 11, 1)
# bits to simulate = 100/ expected BER
num_bits = 1_000_000  # 1 million bits gives reliable errors up to 10 dB!
ber_simulated = []

for eb_no in eb_no_dbs:
    tx_bits = np.random.randint(0,2,num_bits)
    symbols = qpsk_modulate(tx_bits)
    noisy_symbols = add_noise(symbols,eb_no)
    rx_bits = qpsk_demodulate(noisy_symbols)
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    ber_simulated.append(ber)

eb_no_linear = 10 ** (eb_no_dbs / 10)
ber_theory = 0.5 * erfc(np.sqrt(eb_no_linear))

plt.figure(figsize=(10,6))

plt.semilogy(eb_no_dbs,ber_theory,'b-',label='Theoretical QPSK')
plt.semilogy(eb_no_dbs,ber_simulated,'ro',label='Simulated QPSK')

plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel('Eb/N0 (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('QPSK BER vs. Eb/N0 Performance over AWGN Channel', fontsize=14)
plt.legend(fontsize=12)
plt.ylim([1e-6, 1])

plt.savefig('qpsk_ber_curve.png', dpi=300)
plt.show()