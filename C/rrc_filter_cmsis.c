#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "test_vectors.h"

typedef int16_t q15_t;  // Q15 data
typedef int32_t q31_t;  // For 16-bit x 16-bit products
typedef int64_t q63_t;  // The Accumulator

typedef struct {
    uint16_t numTaps;       // Number of filter coefficients 
    q15_t *pState;          // Pointer to the delay-line (history buffer)
    const q15_t *pCoeffs;   // Pointer to our RRC taps array
} arm_fir_instance_q15;

void arm_fir_init_q15(
    arm_fir_instance_q15 * S,
    uint16_t numTaps,
    const q15_t * pCoeffs,
    q15_t * pState,
    uint32_t blockSize
) {
    S->numTaps = numTaps;
    S->pCoeffs = pCoeffs;
    S->pState = pState;

    // Clear the state buffer to zeros
    memset(pState, 0, (numTaps + blockSize - 1) * sizeof(q15_t));
}

void arm_fir_q15(
    const arm_fir_instance_q15 * S,  
    const q15_t * pSrc,              
    q15_t * pDst,                    
    uint32_t blockSize               
){
    q15_t *pState = S->pState;
    const q15_t *pCoeffs = S->pCoeffs;
    q15_t *pStateCurnt;
    q15_t *px;
    const q15_t *pb;
    uint16_t numTaps = S->numTaps;
    uint32_t blkCnt, tapCnt;
    q63_t acc0;

    pStateCurnt = &(pState[(numTaps - 1U)]);
    memcpy(pStateCurnt, pSrc, blockSize * sizeof(q15_t));

      blkCnt = blockSize; // blkCnt = 64
    while (blkCnt > 0U) {
        acc0 = 0;           // Reset the 64-bit sum to zero
        px = pStateCurnt;   // px starts at the current sample
        pb = pCoeffs;       // pb starts at tap h[0]
        tapCnt = numTaps;   // tapCnt = 31 taps

        while (tapCnt > 0U) {
            acc0 += (q63_t)((q31_t)*px-- * *pb++);
            tapCnt--;
        }
                // 1. Shift Q30 down to Q15 (shift right by 15 bits)
        q31_t out = (q31_t)(acc0 >> 15);

        // 2. Saturation (Clamping)
        if (out > 32767) out = 32767;
        if (out < -32768) out = -32768;

        // 3. Store result in output array
        *pDst++ = (q15_t)out;

        // 4. Slide current sample pointer forward by 1 for the next output
        pStateCurnt++;
        blkCnt--;
    }
        memmove(pState, &pState[blockSize], (numTaps - 1U) * sizeof(q15_t));
}


int main(){
    arm_fir_instance_q15 S;
    q15_t state[NUM_TAPS + BLOCK_SIZE - 1];
    q15_t output_c[BLOCK_SIZE];
    // 1. Initialize the filter
    arm_fir_init_q15(&S, NUM_TAPS, rrc_taps_q15, state, BLOCK_SIZE);
        // 2. Run the filter on the test vectors
    arm_fir_q15(&S, test_input_q15, output_c, BLOCK_SIZE);
    // 3. Quantitative Verification against Python Golden Model
    int max_err = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        int diff = abs(output_c[i] - expected_output_q15[i]);
        if (diff > max_err) {
            max_err = diff;
        }
    }
    printf("===================================================\n");
    printf("   ARM CMSIS-DSP RRC FILTER VERIFICATION (C vs PYTHON) \n");
    printf("===================================================\n");
    printf("Taps: %d | Block Size: %d samples\n", NUM_TAPS, BLOCK_SIZE);
    printf("Maximum Difference vs Python Golden Model: %d LSB(s)\n", max_err);
    if (max_err <= 1) {
        printf("VERIFICATION STATUS: >>> PASSED <<< (Bit-Accurate!)\n");
    } else {
        printf("VERIFICATION STATUS: >>> FAILED <<<\n");
    }
    printf("===================================================\n");
    return 0;
}