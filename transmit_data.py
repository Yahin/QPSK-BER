import numpy as np
from qpsk_modem import qpsk_modulate, add_noise, qpsk_demodulate

message = "Cubesat Telemetry: Batt=98%, Temp=22c, Status=NOMINAL"

byte_data = np.frombuffer(message.encode('utf-8'), dtype=np.uint8)
tx_bits = np.unpackbits(byte_data)

eb_no = 10
symbols = qpsk_modulate(tx_bits)
noisy_symbols = add_noise(symbols,eb_no)
rx_bits = qpsk_demodulate(noisy_symbols)

rx_bytes = np.packbits(rx_bits)
recovered_message = bytes(rx_bytes).decode('utf-8', errors='replace')
print(f"Message:{recovered_message}")