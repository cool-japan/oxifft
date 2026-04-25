/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_wasmfft_free: (a: number, b: number) => void;
export const fft_f32: (a: number, b: number, c: number, d: number) => [number, number];
export const fft_f64: (a: number, b: number, c: number, d: number) => [number, number];
export const ifft_f32: (a: number, b: number, c: number, d: number) => [number, number];
export const ifft_f64: (a: number, b: number, c: number, d: number) => [number, number];
export const rfft_f64: (a: number, b: number) => [number, number];
export const wasmfft_forward: (a: number, b: number, c: number, d: number, e: number) => [number, number];
export const wasmfft_forward_interleaved: (a: number, b: number, c: number) => [number, number];
export const wasmfft_inverse: (a: number, b: number, c: number, d: number, e: number) => [number, number];
export const wasmfft_inverse_interleaved: (a: number, b: number, c: number) => [number, number];
export const wasmfft_new: (a: number) => number;
export const wasmfft_size: (a: number) => number;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
