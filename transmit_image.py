import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from qpsk_modem import qpsk_modulate,add_noise, qpsk_demodulate

def create_synthetic_satellite_image(size=120):
    """Creates a simple 120x120 grayscale test pattern resembling a satellite."""
    img = np.zeros((size,size), dtype=np.uint8)
    
    x = np.linspace(30, 80, size, dtype=np.uint8)
    img[:, :] = x
    
    img[45:75, 45:75] = 230
    
    # Solar Panels (Left and Right wings)
    img[53:67, 15:40] = 180
    img[53:67, 80:105] = 180
    
    # Antenna dish (top)
    img[35:45, 55:65] = 255
    
    return img




if __name__ == "__main__":
    img = Image.open("Sage.png").convert("RGB") # change the image here
    img.resize((256,256))
    original_img = np.array(img,dtype=np.uint8)
    #original_img = create_synthetic_satellite_image()
    # H,W = original_img.shape

    flat_bytes = original_img.flatten()
    tx_bits = np.unpackbits(flat_bytes)

    symbols = qpsk_modulate(tx_bits)
    eb_no = 0.001
    noisy_symbols = add_noise(symbols, eb_no)
    rx_bits = qpsk_demodulate(noisy_symbols)
        

    rx_bytes = np.packbits(rx_bits)
    rx_img = rx_bytes.reshape(original_img.shape)

    fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Transmitted", fontsize=12)
    axes[0].axis('off')


    axes[1].imshow(rx_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"Received @ {eb_no} dB\n(BER ~ {np.sum(tx_bits != np.unpackbits(rx_img.flatten()))/len(tx_bits):.1e})", fontsize=11)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()