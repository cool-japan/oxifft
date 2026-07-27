use oxifft_codegen::gen_multi_transform_codelet;

gen_multi_transform_codelet!(size = 3, v = 8, isa = avx2, ty = f32);

fn main() {}
