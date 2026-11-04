import numpy as np

def FSPL_db(distance:float, fc:float):
    """
    Computes Free Space Path Loss (FSPL) in decibels.
    
    Formula: FSPL (dB) = 20 * log10(4 * pi * distance * fc / c)
    """
    c = 3*10**8

    fspl_db = 20*np.log10(distance) +  20*np.log10(fc) + 20*np.log10(4*np.pi/c)

    return fspl_db


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. Mission Link Parameters (Explicit Physical Units)
    # -------------------------------------------------------------
    distance_m = 600e3          # Orbital slant range / distance (600 km LEO)
    fc_Hz      = 2.2e9          # Carrier frequency (2.2 GHz S-band)
    ptx_dBW    = 0.0            # TX RF output power (0.0 dBW = 1.0 Watt)
    gtx_dBi    = 3.0            # CubeSat TX antenna gain (dipole/patch)
    grx_dBi    = 18.0           # Ground station antenna gain (dish/array)
    T_sys_K    = 300.0          # Ground station system noise temperature (Kelvin)
    R_b_bps    = 1e6            # Target data rate (1.0 Mbps)
    k_B        = 1.38e-23       # Boltzmann constant (J/K)
    ber_target_ebno_db = 9.6    # Required Eb/N0 for uncoded QPSK @ BER = 1e-5
    # -------------------------------------------------------------
    # 2. Path Loss Calculation & Sanity Check
    # -------------------------------------------------------------
    fspl_db = FSPL_db(distance_m, fc_Hz)
    # Sanity check: 600 km @ 2.2 GHz yields ~154.85 dB
    assert abs(fspl_db - 154.85) < 0.2, f"FSPL sanity assertion failed: got {fspl_db:.2f} dB"
    # -------------------------------------------------------------
    # 3. Link Budget Equations
    # -------------------------------------------------------------
    # Received Power: Prx = Ptx + Gtx - FSPL + Grx
    prx_dBW = ptx_dBW + gtx_dBi - fspl_db + grx_dBi
    prx_dBm = prx_dBW + 30.0
    # Energy per Bit: Eb = Prx - 10*log10(Rb)
    eb_dB = prx_dBW - 10.0 * np.log10(R_b_bps)
    # Noise Spectral Density: N0 = 10*log10(k_B * T_sys)
    n0_dB = 10.0 * np.log10(k_B) + 10.0 * np.log10(T_sys_K)
    # Received Eb/N0 & Margin
    eb_no_db = eb_dB - n0_dB
    margin_db = eb_no_db - ber_target_ebno_db
    # -------------------------------------------------------------
    # 4. Formatted Mission Link Report
    # -------------------------------------------------------------
    print("=" * 55)
    print("   CUBESAT S-BAND LEO DOWNLINK BUDGET ANALYSIS")
    print("=" * 55)
    print(f"Carrier Frequency (fc)     : {fc_Hz / 1e9:.2f} GHz")
    print(f"Link Distance (d)          : {distance_m / 1e3:.1f} km")
    print(f"Free Space Path Loss (FSPL): {fspl_db:.2f} dB")
    print(f"Transmit Power (Ptx)       : {ptx_dBW:.1f} dBW ({10**(ptx_dBW/10):.1f} W)")
    print(f"Antenna Gains (TX / RX)    : +{gtx_dBi:.1f} dBi / +{grx_dBi:.1f} dBi")
    print("-" * 55)
    print(f"Received Power (Prx)       : {prx_dBW:.2f} dBW ({prx_dBm:.2f} dBm)")
    print(f"Data Bit Rate (Rb)         : {R_b_bps / 1e6:.1f} Mbps")
    print(f"Noise Density (N0)         : {n0_dB:.2f} dBW/Hz (Tsys = {T_sys_K:.0f} K)")
    print(f"Received Eb/N0             : {eb_no_db:.2f} dB")
    print(f"Required Eb/N0 (@ 1e-5 BER): {ber_target_ebno_db:.2f} dB")
    print("-" * 55)
    status_str = "CLOSED (PASSED)" if margin_db >= 0 else "DEFICIT (FAILED)"
    print(f"Link Margin                : {margin_db:+.2f} dB  -->  {status_str}")
    print("=" * 55)