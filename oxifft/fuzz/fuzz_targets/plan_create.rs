#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Cap at 4097 to keep each iteration fast even for prime-sized DCT/DST
    let n = n % 4097;
    let dir_byte = data[4] & 1;
    let kind_byte = data[5] % 9;

    use oxifft::{Direction, Flags, Plan, R2rPlan, RealPlan};

    let direction = if dir_byte == 0 {
        Direction::Forward
    } else {
        Direction::Backward
    };

    let _ = Plan::<f64>::dft_1d(n, direction, Flags::ESTIMATE);
    let _ = RealPlan::<f64>::r2c_1d(n, Flags::ESTIMATE);

    // R2rKind: map kind_byte 0..=8 to the 9 variants
    // We use the convenience constructors which are always public
    let _ = match kind_byte {
        0 => R2rPlan::<f64>::dct1(n, Flags::ESTIMATE),
        1 => R2rPlan::<f64>::dct2(n, Flags::ESTIMATE),
        2 => R2rPlan::<f64>::dct3(n, Flags::ESTIMATE),
        3 => R2rPlan::<f64>::dct4(n, Flags::ESTIMATE),
        4 => R2rPlan::<f64>::dst1(n, Flags::ESTIMATE),
        5 => R2rPlan::<f64>::dst2(n, Flags::ESTIMATE),
        6 => R2rPlan::<f64>::dst3(n, Flags::ESTIMATE),
        7 => R2rPlan::<f64>::dst4(n, Flags::ESTIMATE),
        _ => R2rPlan::<f64>::dht(n, Flags::ESTIMATE),
    };
});
