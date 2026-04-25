/* @ts-self-types="./oxifft.d.ts" */

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
class WasmFft {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmFftFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmfft_free(ptr, 0);
    }
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
     * @param {Float64Array} real
     * @param {Float64Array} imag
     * @returns {Float64Array}
     */
    forward(real, imag) {
        const ptr0 = passArrayF64ToWasm0(real, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(imag, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmfft_forward(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v3;
    }
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
     * @param {Float64Array} interleaved
     * @returns {Float64Array}
     */
    forward_interleaved(interleaved) {
        const ptr0 = passArrayF64ToWasm0(interleaved, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmfft_forward_interleaved(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
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
     * @param {Float64Array} real
     * @param {Float64Array} imag
     * @returns {Float64Array}
     */
    inverse(real, imag) {
        const ptr0 = passArrayF64ToWasm0(real, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(imag, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmfft_inverse(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v3;
    }
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
     * @param {Float64Array} interleaved
     * @returns {Float64Array}
     */
    inverse_interleaved(interleaved) {
        const ptr0 = passArrayF64ToWasm0(interleaved, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmfft_inverse_interleaved(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * Create a new WASM FFT wrapper.
     *
     * # Arguments
     *
     * * `size` - Transform size
     * @param {number} size
     */
    constructor(size) {
        const ret = wasm.wasmfft_new(size);
        this.__wbg_ptr = ret >>> 0;
        WasmFftFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the transform size.
     * @returns {number}
     */
    get size() {
        const ret = wasm.wasmfft_size(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmFft.prototype[Symbol.dispose] = WasmFft.prototype.free;
exports.WasmFft = WasmFft;

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
 * @param {Float32Array} real
 * @param {Float32Array} imag
 * @returns {Float32Array}
 */
function fft_f32(real, imag) {
    const ptr0 = passArrayF32ToWasm0(real, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(imag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fft_f32(ptr0, len0, ptr1, len1);
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}
exports.fft_f32 = fft_f32;

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
 * @param {Float64Array} real
 * @param {Float64Array} imag
 * @returns {Float64Array}
 */
function fft_f64(real, imag) {
    const ptr0 = passArrayF64ToWasm0(real, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(imag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.fft_f64(ptr0, len0, ptr1, len1);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}
exports.fft_f64 = fft_f64;

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
 * @param {Float32Array} real
 * @param {Float32Array} imag
 * @returns {Float32Array}
 */
function ifft_f32(real, imag) {
    const ptr0 = passArrayF32ToWasm0(real, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(imag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ifft_f32(ptr0, len0, ptr1, len1);
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}
exports.ifft_f32 = ifft_f32;

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
 * @param {Float64Array} real
 * @param {Float64Array} imag
 * @returns {Float64Array}
 */
function ifft_f64(real, imag) {
    const ptr0 = passArrayF64ToWasm0(real, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(imag, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.ifft_f64(ptr0, len0, ptr1, len1);
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}
exports.ifft_f64 = ifft_f64;

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
 * @param {Float64Array} input
 * @returns {Float64Array}
 */
function rfft_f64(input) {
    const ptr0 = passArrayF64ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rfft_f64(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}
exports.rfft_f64 = rfft_f64;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_6b64449b9b9ed33c: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./oxifft_bg.js": import0,
    };
}

const WasmFftFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmfft_free(ptr >>> 0, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
function decodeText(ptr, len) {
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

const wasmPath = `${__dirname}/oxifft_bg.wasm`;
const wasmBytes = require('fs').readFileSync(wasmPath);
const wasmModule = new WebAssembly.Module(wasmBytes);
let wasm = new WebAssembly.Instance(wasmModule, __wbg_get_imports()).exports;
wasm.__wbindgen_start();
