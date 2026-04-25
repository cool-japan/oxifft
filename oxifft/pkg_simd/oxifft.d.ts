/* tslint:disable */
/* eslint-disable */

/**
 * WebAssembly FFT wrapper for JavaScript.
 *
 * Provides a plan-based interface for repeated FFT operations.
 *
 * # Example (JavaScript)
 *
 * ```javascript
 * import { WasmFft } from 'oxifft';
 *
 * const fft = new WasmFft(256);
 * const real = new Float64Array([1, 2, 3, ...]);
 * const imag = new Float64Array([0, 0, 0, ...]);
 * const result = fft.forward(real, imag);
 * ```
 */
export class WasmFft {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Execute forward FFT.
     *
     * # Arguments
     *
     * * `real` - Real part of input (length must equal size)
     * * `imag` - Imaginary part of input (length must equal size)
     *
     * # Returns
     *
     * Interleaved output: [re0, im0, re1, im1, ...]
     */
    forward(real: Float64Array, imag: Float64Array): Float64Array;
    /**
     * Execute forward FFT with interleaved input.
     *
     * # Arguments
     *
     * * `interleaved` - Interleaved complex input [re0, im0, re1, im1, ...]
     *
     * # Returns
     *
     * Interleaved output.
     */
    forward_interleaved(interleaved: Float64Array): Float64Array;
    /**
     * Execute inverse FFT.
     *
     * # Arguments
     *
     * * `real` - Real part of input (length must equal size)
     * * `imag` - Imaginary part of input (length must equal size)
     *
     * # Returns
     *
     * Interleaved output: [re0, im0, re1, im1, ...]
     */
    inverse(real: Float64Array, imag: Float64Array): Float64Array;
    /**
     * Execute inverse FFT with interleaved input.
     *
     * # Arguments
     *
     * * `interleaved` - Interleaved complex input [re0, im0, re1, im1, ...]
     *
     * # Returns
     *
     * Interleaved output (normalized).
     */
    inverse_interleaved(interleaved: Float64Array): Float64Array;
    /**
     * Create a new WASM FFT wrapper.
     *
     * # Arguments
     *
     * * `size` - Transform size
     */
    constructor(size: number);
    /**
     * Get the transform size.
     */
    readonly size: number;
}

/**
 * One-shot forward FFT (f32).
 *
 * # Arguments
 *
 * * `real` - Real part of input
 * * `imag` - Imaginary part of input
 *
 * # Returns
 *
 * Interleaved output: [re0, im0, re1, im1, ...]
 */
export function fft_f32(real: Float32Array, imag: Float32Array): Float32Array;

/**
 * One-shot forward FFT (f64).
 *
 * # Arguments
 *
 * * `real` - Real part of input
 * * `imag` - Imaginary part of input
 *
 * # Returns
 *
 * Interleaved output: [re0, im0, re1, im1, ...]
 */
export function fft_f64(real: Float64Array, imag: Float64Array): Float64Array;

/**
 * One-shot inverse FFT (f32).
 *
 * # Arguments
 *
 * * `real` - Real part of input
 * * `imag` - Imaginary part of input
 *
 * # Returns
 *
 * Interleaved output: [re0, im0, re1, im1, ...] (normalized)
 */
export function ifft_f32(real: Float32Array, imag: Float32Array): Float32Array;

/**
 * One-shot inverse FFT (f64).
 *
 * # Arguments
 *
 * * `real` - Real part of input
 * * `imag` - Imaginary part of input
 *
 * # Returns
 *
 * Interleaved output: [re0, im0, re1, im1, ...] (normalized)
 */
export function ifft_f64(real: Float64Array, imag: Float64Array): Float64Array;

/**
 * Real-to-complex FFT (f64).
 *
 * # Arguments
 *
 * * `input` - Real input signal
 *
 * # Returns
 *
 * Interleaved complex output (hermitian symmetric, n/2+1 complex values)
 */
export function rfft_f64(input: Float64Array): Float64Array;
