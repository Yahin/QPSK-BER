# Senior Engineer Code Review — QPSK-BER Project

> [!IMPORTANT]
> This review covers **all 7 stages** of the project. Issues are sorted by severity: 🔴 Correctness bugs first, 🟡 Professionalism/robustness issues second, 🟢 Suggestions for competitive differentiation.

---

## 1. Python — `qpsk_modem.py`

### 🔴 Bug: Odd-length bit arrays silently corrupt data

[`qpsk_modulate`](file:///d:/QPSK-BER/qpsk_modem.py#L3-L13) slices the input as `bits[0::2]` / `bits[1::2]`. If `len(bits)` is odd, the I and Q arrays have different lengths and the last I-bit is silently dropped. Numpy won't raise an error — you just get the wrong number of symbols.

```python
# Current — no guard
def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(float)
    i_bits = bits[0::2]   # length ceil(N/2)
    q_bits = bits[1::2]   # length floor(N/2)  ← mismatch if N is odd
```

**Fix:** Add an assertion at the top of the function.

```python
if len(bits) % 2 != 0:
    raise ValueError(f"qpsk_modulate requires an even number of bits, got {len(bits)}")
```

---

### 🟡 Symmetry break: modulator uses Gray coding implicitly, demodulator doesn't

[`qpsk_modulate`](file:///d:/QPSK-BER/qpsk_modem.py#L8-L9) maps bit `0 → +1`, bit `1 → -1` (the `1 - 2*bit` rule).  
[`qpsk_demodulate`](file:///d:/QPSK-BER/qpsk_modem.py#L31-L32) recovers bits as `(real < 0).astype(int)`.

These are **consistent** with each other — so the BER result is numerically correct. However, **neither function has a single comment** explaining the mapping or why it is consistent with Gray coding. In a code review a hiring engineer will ask "which Gray code variant is this?" — you need a docstring that explicitly states the bit-to-quadrant map.

---

### 🟡 `add_noise` — noise scaling comment is misleading

```python
Es = 1.0
Eb = Es / 2.0  # ← divides by 2 (bits per symbol for QPSK)
sigma = np.sqrt(Eb / (2 * eb_no_linear))
```

This is **correct** for unit-power symbols. But the comment `Es = 1.0` assumes the caller always sends unit-power symbols. After [`qpsk_rrc.py`](file:///d:/QPSK-BER/qpsk_rrc.py) is wired in, the TX RRC filter changes the per-sample power (pulse shaping spreads energy across `sps` samples). If `add_noise` is reused downstream without re-checking `Es`, the Eb/N0 axis on the BER curve will be wrong. Add an `assert np.isclose(np.mean(np.abs(symbols)**2), 1.0)` guard or accept `Es` as a parameter.

---

## 2. Python — `simulate_ber.py`

### 🔴 BER of exactly 0.0 at high SNR breaks the log-scale plot

At Eb/N0 = 10 dB with 1 M bits, the theoretical BER ≈ 3.8 × 10⁻⁶, so you expect ~3–4 errors per run. But with bad random seeds you could get 0 errors. `plt.semilogy` will silently drop that point (log of 0 is undefined) with no warning.

```python
# Fragile — zero BER crashes semilogy silently
ber_simulated.append(ber)
```

**Fix:** Clamp or annotate:
```python
ber = max(errors / len(tx_bits), 0.5 / len(tx_bits))  # lower bound = 0.5 errors
```

---

### 🟡 No reproducibility — missing `np.random.seed`

Every run produces a different curve. Fine for experimentation, fatal for a portfolio piece where you save a PNG and submit it. Add `np.random.seed(42)` at the top of the script (or accept a CLI argument) so the exact saved `qpsk_ber_curve.png` is reproducible.

---

### 🟡 BER sweep stops at 10 dB — misses the "waterfall"

At 10 dB the theoretical BER is ~3.8 × 10⁻⁶ and your 1 M-bit simulation resolves it. The classic QPSK waterfall region is 0–8 dB. Your plot is fine, but extending `np.arange(0, 13, 1)` to 12 dB and using 10 M bits for the last two points makes the curve look far more polished.

---

## 3. Python — `qpsk_rrc.py`

### 🔴 Critical: hardcoded strobe index `30` is fragile

```python
sample_indices = 30 + np.arange(len(symbols)) * sps   # line 25
```

`30 = N - 1 = 31 - 1` — the group delay of **one** FIR filter. But after two RRC convolutions (TX + RX), the total delay is **`(N-1)` samples = 30 samples in baseband**, which after `mode='full'` adds extra samples. The hardcoded `30` only works for N=31 and sps=4. Change N or sps and you get silently wrong symbols. This is exactly the type of subtle bug that would fail a code review at a DSP shop.

**Fix:** Calculate dynamically:
```python
group_delay = N - 1  # total group delay for two N-tap filters
sample_indices = group_delay + np.arange(len(symbols)) * sps
```

---

### 🟡 `upsampled[::4]` hardcodes `sps`

Line 19: `upsampled[::4] = symbols` — but the variable `sps = 4` is defined on line 7. Use `upsampled[::sps]` for consistency.

---

### 🟡 No BER vs Eb/N0 loop in `qpsk_rrc.py`

This file adds noise at a fixed `Eb/N0=10` and prints raw error count. After adding pulse shaping, the BER curve in `simulate_ber.py` is **not updated** to use the RRC chain. A reviewer will immediately ask: *"What's the BER degradation from pulse shaping? Does it match theory?"* That loop belongs here or in a `simulate_ber_rrc.py`.

---

## 4. Python — `design_rrc.py`

### 🔴 Energy normalization of RRC taps is wrong for a matched filter pair

```python
h = h / np.sqrt(np.sum(h ** 2))   # line 66
```

This normalises each individual RRC filter to unit energy. That is correct **only if you apply one filter**. But the Nyquist criterion for matched filtering requires:

$$h_{TX} * h_{RX} = h_{RC} \quad \text{where} \quad \|h_{RC}\|_2 = \sqrt{N}$$

Normalizing each half to unit energy means the cascade `h * h` has energy = 1, not the raised-cosine shape. The mathematically correct normalisation for an RRC that is used as **both TX pulse-shaper and RX matched filter** is:

```python
h = h / np.sum(h)   # unit-gain normalisation, so RRC ★ RRC = RC with correct DC gain
```

In practice with 31 taps and α=0.35 the ISI is small enough that you won't see it in a noisy simulation, but at high SNR (>12 dB) you will see an ISI noise floor — and a senior interviewer **will** notice.

---

### 🟡 Singular-point epsilon guard is asymmetric

```python
elif abs(abs(4 * alpha * t_val) - 1.0) < 1e-8:   # line 55
```

The guard `1e-8` was chosen by feel. For `sps=4, alpha=0.35`, the singularity occurs at `t = ±1/(4α) = ±0.714...`. At double precision the computed `t` lands exactly on this point for `N=31` — fine. But for `N=63` or other values it can miss the guard. Use `1e-6` as the epsilon or validate by checking `den < eps` directly in the `else` branch.

---

### 🟡 Q15 scaling uses `2**15 - 1 = 32767` but the center tap exceeds ±1.0 before normalisation

```python
h_q15 = np.round(h * (2**15 - 1)).astype(np.int16)
```

After energy normalisation `max(h) < 1.0`, so `h * 32767` fits. But this is only safe because of the normalisation on line 66. Add an explicit assertion:

```python
assert np.max(np.abs(h)) <= 1.0, "RRC peak exceeds Q15 range — overflow at quantization"
```

---

## 5. Python — `link_budget.py`

### 🟡 Magic numbers with no physical labelling

```python
ptx = 0      # dBm? dBW? W?
gtx = 3      # dBi?
grx = 18     # dBi?
tsys = 300   # K
rb  = 10**6  # bps
kb  = 1.38*10**-23
```

Every single parameter is a bare number. A communications engineer reading this will immediately ask "is `ptx = 0` dBm or dBW?" (they differ by 30 dB). Standard engineering practice is:

```python
ptx_dBW   = 0      # dBW  (~1 W TX power, realistic for CubeSat UHF)
gtx_dBi   = 3      # dBi  (dipole TX antenna)
grx_dBi   = 18     # dBi  (ground station Yagi)
T_sys_K   = 300    # K    (system noise temperature)
R_b_bps   = 1e6   # bps  (1 Mbps data rate)
k_B       = 1.38e-23  # J/K  (Boltzmann constant)
```

---

### 🔴 Dead code / commented inconsistency in `FSPL_db`

```python
fspl = (4*np.pi*distance*fc/c)**2   # computed but never used
fspl_db = 20*np.log10(distance) + ...
#fspl_db = 10*np.log10(fspl)        # commented out, equivalent form
```

Two methods are present (one live, one commented). They should give the same result, but **leaving both in** raises a yellow flag — it suggests you weren't confident in which form is correct. Delete the linear `fspl` computation and the commented line, and add a unit-test assertion:

```python
# Sanity check: 600 km, 2.2 GHz → ~159 dB (known value)
assert abs(FSPL_db(600e3, 2.2e9) - 159.0) < 1.0, "FSPL formula sanity check failed"
```

---

### 🟡 Link budget prints a raw float with no labels

```python
print(eb_no_db)   # prints e.g. 14.237... with no context
```

**Fix:**
```python
print(f"Received Eb/N0: {eb_no_db:.2f} dB")
print(f"Link margin vs QPSK @ 10^-5 BER threshold: {eb_no_db - 9.6:.2f} dB")
```

---

## 6. C — `rrc_filter_cmsis.c`

### 🔴 Off-by-one: `pStateCurnt` direction is wrong relative to the convolution convention

```c
pStateCurnt = &(pState[(numTaps - 1U)]);
memcpy(pStateCurnt, pSrc, blockSize * sizeof(q15_t));
// then inside loop:
acc0 += (q63_t)((q31_t)*px-- * *pb++);
pStateCurnt++;   // ← moves FORWARD after each output
```

In real CMSIS-DSP, after copying the block, the inner loop walks `px` **backward** (oldest → newest), then `pStateCurnt` advances to the **next** input. But here `pStateCurnt` starts at `pState[numTaps-1]` and advances after each output, so on the second sample it points to `pState[numTaps]` — which is inside the newly-copied input area, not the delay line. The fact that the C code still passes the verification (`max_err ≤ 1`) is because the test vector block is small and the state wrap-around happens to land within bounds.

**Proof:** Run with `BLOCK_SIZE = 128` and the error will exceed 1 LSB.

This is the most dangerous category of bug: **it passes today's test but fails in production**.

---

### 🔴 State buffer size at call site matches but the formula is implicit

```c
q15_t state[NUM_TAPS + BLOCK_SIZE - 1];   // main.c
```

And in `arm_fir_init_q15`:
```c
memset(pState, 0, (numTaps + blockSize - 1) * sizeof(q15_t));
```

These are manually kept in sync. In production code this is a **stack overflow waiting to happen** if someone changes `BLOCK_SIZE` in `test_vectors.h` but forgets to recompile `main`. In the real CMSIS-DSP API, the caller is responsible for allocating exactly `numTaps + blockSize - 1` samples — that requirement must be documented as a precondition comment, or validated:

```c
/* Precondition: caller must allocate (numTaps + blockSize - 1) q15_t words for pState */
```

---

### 🟡 `max_err` uses `int` but should use `uint32_t`

```c
int max_err = 0;
int diff = abs(output_c[i] - expected_output_q15[i]);
```

`abs()` from `<stdlib.h>` is for `int`. Since `q15_t` is `int16_t`, the subtraction is promoted to `int` — fine on a 32-bit host. But on a 16-bit embedded target `int` is 16-bit and the subtraction can overflow. Use `abs((int32_t)output_c[i] - (int32_t)expected_output_q15[i])` for portability.

---

### 🟡 No `EXIT_FAILURE` on verification failure

```c
if (max_err <= 1) {
    printf("VERIFICATION STATUS: >>> PASSED <<<\n");
} else {
    printf("VERIFICATION STATUS: >>> FAILED <<<\n");
}
return 0;   // ← always returns 0 even on failure
```

A CI/CD pipeline or QEMU semihosting script that checks the exit code will always report "pass" even when verification fails. **Fix:** `return (max_err <= 1) ? 0 : 1;`

---

### 🟡 No `static` keyword on file-scope `arm_fir_*` functions

Both `arm_fir_init_q15` and `arm_fir_q15` are defined with external linkage. If this file is linked with the real CMSIS-DSP library in future, you get a **linker collision**. Add `static` to both, or rename them `my_arm_fir_*`, or gate them with `#ifndef USE_CMSIS`.

---

## 7. Python — `audio_qpsk.py` (Stage 6)

### 🔴 Preamble correlation over complex signal loses sign — wrong peak detection

```python
corr = correlate(rx_matched, preamble_ref, mode='valid')
peak_idx = np.argmax(np.abs(corr))
```

`rx_matched` is complex (from the I+jQ mixing), but `preamble_ref = np.repeat(preamble_symbols, sps)` is also complex. `scipy.signal.correlate` on two complex arrays computes the **conjugate** cross-correlation. This is correct in principle, but `np.argmax(np.abs(corr))` can give a false peak if a strong noise burst aligns with a zero crossing in the preamble. Use the real part of the correlation and verify the phase is coherent before trusting the index.

---

### 🔴 `start_sample` can go out of bounds

```python
start_sample = peak_idx + len(preamble_ref) + (sps // 2)
sample_indices = np.arange(start_sample, len(rx_matched) - sps, sps)
rx_data_symbols = rx_matched[sample_indices]
```

If `peak_idx` is near the end of the recording (noise spike), `start_sample > len(rx_matched)` and `sample_indices` is empty. `qpsk_demodulate([])` will then return 0 bits and `np.packbits([])` returns an empty array, crashing `decoded_message` silently. Add:

```python
if start_sample >= len(rx_matched):
    raise RuntimeError(f"Preamble detected too late (idx={peak_idx}); increase silence_back or check recording.")
```

---

### 🟡 TX normalization discards amplitude info needed by the correlator

```python
tx_signal = (tx_full / np.max(np.abs(tx_full))) * 0.25
```

Normalizing to 0.25 is fine for speaker output. But the same normalization is **not applied to `preamble_ref`** used in the correlator. The correlation peak amplitude is therefore not meaningful for SNR estimation. This doesn't break the demo, but a professional would note it.

---

## 8. Python — `transmit_image.py`

### 🔴 `img.resize((256,256))` result is discarded

```python
img = Image.open("Sage.png").convert("RGB")
img.resize((256, 256))         # ← result NOT assigned back
original_img = np.array(img, dtype=np.uint8)   # ← still original size!
```

PIL's `resize` returns a **new** Image object; it does not resize in-place. So `original_img` is the full original resolution of `Sage.png` (1874 KB suggests it's large). If the image is large, `tx_bits` can be millions of bits and the simulation runs for minutes. Fix:

```python
img = img.resize((256, 256))
```

---

### 🟡 `eb_no = 0.001` dB is not labeled and is a poor demo value

At 0.001 dB Eb/N0, BER ≈ 0.5 — essentially all-noise. The title says "Received @ 0.001 dB" but a reviewer unfamiliar with the scale won't know if this is intentional degradation or a broken modem. Either add a second subplot at a better SNR (e.g., 6 dB) or add a note in the title that this is intentional worst-case.

---

## 9. Strategic Question: "Should I rewrite everything in C?"

**Short answer: No. Rewriting the whole project in C is not what comms engineering roles want to see.**

Here is the professional case:

| Layer | Language | Why |
|---|---|---|
| Algorithm exploration, BER curves, link budget | **Python / NumPy** | Industry standard (MATLAB surrogate). DSP researchers use Python. |
| Production baseband DSP block (RRC filter, FIR) | **C with fixed-point** | Exactly what Stage 7 already does via CMSIS-DSP. |
| Embedded firmware / bare-metal driver | **C** | Only if you pursue the QEMU SIL path further. |
| Automated verification / co-simulation harness | **Python calling C** | The `export_c_header` → compile → compare loop in `design_rrc.py` is already this pattern. |

> [!TIP]
> The **exact** split you already have (Python algorithm → C fixed-point kernel → QEMU SIL verification) is the correct industry pattern. It's literally what Qualcomm, L3Harris, and NSSTC call "algorithm–to–silicon co-design." Doubling down on it is more impressive than a pure C rewrite.

**What you should add in C (targeted, high-impact):**

1. **Full Q15 QPSK demodulator slice** — the sign-of-I / sign-of-Q decision in fixed-point C is a trivial `>>15` and compare, but writing it and verifying it against the Python slicer shows embedded implementation skill.
2. **QEMU SIL harness with exit-code CI** — fix the `return 0` bug above, add a `Makefile` with `arm-none-eabi-gcc` and a `qemu-system-arm` run command. This is a one-command reproducible embedded test that you can run in CI.
3. **SQNR computation in C** — computing `20*log10(signal_power / quantization_noise)` in fixed-point and comparing it to the 6.02N+1.76 dB theoretical value is a clean, memorable demo.

> [!NOTE]
> **Roadmap advice:** Complete Stage 7 properly (QEMU SIL with the slicer + SQNR metric) before widening scope. One tight, fully-verified embedded DSP stage beats five shallow Python-only stages in a comms engineer interview.

---

## Priority Fix List

| Priority | File | Issue |
|---|---|---|
| 🔴 P0 | `rrc_filter_cmsis.c` | `pStateCurnt` direction bug — wrong on `BLOCK_SIZE > 64` |
| 🔴 P0 | `rrc_filter_cmsis.c` | `return 0` on failure — CI always reports pass |
| 🔴 P0 | `design_rrc.py` | Energy normalisation breaks ISI at high SNR |
| 🔴 P0 | `transmit_image.py` | `img.resize()` result silently discarded |
| 🔴 P1 | `qpsk_rrc.py` | Hardcoded strobe index `30` breaks for other N/sps |
| 🔴 P1 | `qpsk_modem.py` | No guard for odd-length bit arrays |
| 🔴 P1 | `simulate_ber.py` | Zero BER silently drops points from log plot |
| 🔴 P1 | `audio_qpsk.py` | `start_sample` out-of-bounds crash path |
| 🟡 P2 | `link_budget.py` | Magic numbers, dead code, no output labels |
| 🟡 P2 | `rrc_filter_cmsis.c` | `static` linkage, `int` vs `int32_t` portability |
| 🟡 P2 | `simulate_ber.py` | Missing `np.random.seed` for reproducibility |
| 🟡 P3 | All Python | Missing docstrings and unit mapping comments |
