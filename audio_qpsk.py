import numpy as np
import sounddevice as sd
from scipy.signal import correlate
from qpsk_modem import qpsk_modulate,qpsk_demodulate

Fs = 48000              # Audio sampling rate
fc = 3000               # Carrier frequency (2 kHz tone)
baud_rate = 200         # Baud rate for transmission
sps = int(Fs/baud_rate)        # Samples per symbol 

message = "Jesus will help \x00"
byte_data = np.frombuffer(message.encode('utf-8'), dtype=np.uint8)
tx_bits = np.unpackbits(byte_data)
data_symbols = qpsk_modulate(tx_bits)

preamble_bits = np.array([0,0, 0,1, 1,1, 0,0, 1,0, 1,1, 0,1, 1,0], dtype=float)
preamble_symbols = qpsk_modulate(preamble_bits)

all_symbols  = np.concatenate([preamble_symbols,data_symbols])

i_wave = np.repeat(np.real(all_symbols),sps)
q_wave = np.repeat(np.imag(all_symbols),sps)

tx_samples = len(i_wave)
t = np.arange(tx_samples) / Fs

tx_audio = i_wave * np.cos(2*np.pi*fc*t) - q_wave * np.sin(2*np.pi*fc*t)

silence_front = np.zeros(int(0.3*Fs))
silence_back = np.zeros(int(1.2 * Fs))
tx_full = np.concatenate([silence_front,tx_audio,silence_back])

# Normalize volume to 70% to avoid clipping
tx_signal = (tx_full/np.max(np.abs(tx_full))) * 0.7

#sd.play(tx_signal,Fs)
#sd.wait()

print("Transmitting and listening through microphone...")
rx_full = sd.playrec(tx_signal, samplerate=Fs, channels=1, blocking=True).flatten()
print("Done! Recorded", len(rx_full), "samples from physical air.")

t_rx = np.arange(len(rx_full)) / Fs
rx_i = rx_full * (2.0 * np.cos(2 * np.pi * fc * t_rx))
rx_q = rx_full * (-2.0 * np.sin(2 * np.pi * fc * t_rx))
rx = rx_i + 1j * rx_q 

h = np.ones(sps) / sps
rx_matched = np.convolve(rx,h,mode='same')

preamble_ref = np.repeat(preamble_symbols,sps)

corr = correlate(rx_matched, preamble_ref, mode='valid')

peak_idx = np.argmax(np.abs(corr))
phase_offset = np.angle(corr[peak_idx])

start_sample = peak_idx + len(preamble_ref) + (sps // 2)
sample_indices = np.arange(start_sample, len(rx_matched) - sps, sps)
rx_data_symbols = rx_matched[sample_indices]*np.exp(-1j*phase_offset)

rx_bits = qpsk_demodulate(rx_data_symbols)
rx_bytes = np.packbits(rx_bits)
decoded_message = bytes(rx_bytes).decode('utf-8', errors='replace')
clean_text = decoded_message.split('\x00')[0]

print("Max microphone amplitude:", np.max(np.abs(rx_full)))
print("Detected Preamble at sample index:", peak_idx)
print("Detected Phase Rotation:", np.degrees(phase_offset))
print("Decoded Message:",clean_text)