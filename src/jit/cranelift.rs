use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::jit::{CompiledKernel, JitBackend, TileKernelFn};
use crate::kernel_ir::KernelIr;

pub struct CraneliftBackend;

struct CraneliftKernel {
    _module: JITModule,
    fn_ptr: TileKernelFn,
}

// Safety: the JITModule owns the executable memory and keeps it alive.
// The function pointer is valid for the lifetime of the module.
unsafe impl Send for CraneliftKernel {}
unsafe impl Sync for CraneliftKernel {}

impl CompiledKernel for CraneliftKernel {
    fn function_ptr(&self) -> TileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for CraneliftBackend {
    fn compile(&self, ir: &KernelIr) -> Box<dyn CompiledKernel> {
        match ir {
            KernelIr::Mandelbrot { max_iter } => Box::new(compile_mandelbrot(*max_iter)),
        }
    }
}

fn compile_mandelbrot(max_iter: u32) -> CraneliftKernel {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_codegen::isa::lookup_by_name(
        &target_lexicon::Triple::host().to_string(),
    )
    .unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();

    let mut module =
        JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));

    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(I64)); // output: *mut u32
    sig.params.push(AbiParam::new(I32)); // width: u32
    sig.params.push(AbiParam::new(I32)); // height: u32
    sig.params.push(AbiParam::new(F64)); // x_min: f64
    sig.params.push(AbiParam::new(F64)); // y_min: f64
    sig.params.push(AbiParam::new(F64)); // x_step: f64
    sig.params.push(AbiParam::new(F64)); // y_step: f64
    sig.params.push(AbiParam::new(I32)); // row_start: u32
    sig.params.push(AbiParam::new(I32)); // row_end: u32

    let func_id = module
        .declare_function("mandelbrot_tile", Linkage::Export, &sig)
        .unwrap();

    let mut func = Function::with_name_signature(UserFuncName::default(), sig.clone());
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

    // -- Variables --
    let v_output = Variable::from_u32(0);
    let v_width = Variable::from_u32(1);
    let v_x_min = Variable::from_u32(3);
    let v_y_min = Variable::from_u32(4);
    let v_x_step = Variable::from_u32(5);
    let v_y_step = Variable::from_u32(6);
    let v_row_start = Variable::from_u32(7);
    let v_row_end = Variable::from_u32(8);
    let v_row = Variable::from_u32(9);
    let v_col = Variable::from_u32(10);
    let v_zx = Variable::from_u32(11);
    let v_zy = Variable::from_u32(12);
    let v_iter = Variable::from_u32(13);

    builder.declare_var(v_output, I64);
    builder.declare_var(v_width, I32);
    builder.declare_var(v_x_min, F64);
    builder.declare_var(v_y_min, F64);
    builder.declare_var(v_x_step, F64);
    builder.declare_var(v_y_step, F64);
    builder.declare_var(v_row_start, I32);
    builder.declare_var(v_row_end, I32);
    builder.declare_var(v_row, I32);
    builder.declare_var(v_col, I32);
    builder.declare_var(v_zx, F64);
    builder.declare_var(v_zy, F64);
    builder.declare_var(v_iter, I32);

    // -- Blocks --
    let entry_block = builder.create_block();
    let outer_check = builder.create_block();
    let col_init = builder.create_block();
    let inner_check = builder.create_block();
    let mandel_init = builder.create_block();
    let mandel_check = builder.create_block();
    let mandel_body = builder.create_block();
    let mandel_done = builder.create_block();
    let inner_inc = builder.create_block();
    let outer_inc = builder.create_block();
    let exit_block = builder.create_block();

    // -- Entry: read params, init row --
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let params: Vec<_> = builder.block_params(entry_block).to_vec();
    builder.def_var(v_output, params[0]);
    builder.def_var(v_width, params[1]);
    builder.def_var(v_x_min, params[3]);
    builder.def_var(v_y_min, params[4]);
    builder.def_var(v_x_step, params[5]);
    builder.def_var(v_y_step, params[6]);
    builder.def_var(v_row_start, params[7]);
    builder.def_var(v_row_end, params[8]);

    let row_start_val = builder.use_var(v_row_start);
    builder.def_var(v_row, row_start_val);
    builder.ins().jump(outer_check, &[]);

    // -- Outer check: row < row_end --
    builder.switch_to_block(outer_check);
    let row = builder.use_var(v_row);
    let row_end = builder.use_var(v_row_end);
    let done = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, row, row_end);
    builder.ins().brif(done, exit_block, &[], col_init, &[]);

    // -- Col init: col = 0 --
    builder.switch_to_block(col_init);
    builder.seal_block(col_init);
    let zero_i32 = builder.ins().iconst(I32, 0);
    builder.def_var(v_col, zero_i32);
    builder.ins().jump(inner_check, &[]);

    // -- Inner check: col < width --
    builder.switch_to_block(inner_check);
    let col = builder.use_var(v_col);
    let width = builder.use_var(v_width);
    let col_done = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, col, width);
    builder.ins().brif(col_done, outer_inc, &[], mandel_init, &[]);

    // -- Mandel init: zx=0, zy=0, iter=0 --
    builder.switch_to_block(mandel_init);
    builder.seal_block(mandel_init);
    let zero_f64 = builder.ins().f64const(0.0);
    builder.def_var(v_zx, zero_f64);
    let zero_f64_2 = builder.ins().f64const(0.0);
    builder.def_var(v_zy, zero_f64_2);
    let zero_iter = builder.ins().iconst(I32, 0);
    builder.def_var(v_iter, zero_iter);
    builder.ins().jump(mandel_check, &[]);

    // -- Mandel check: escaped? --
    builder.switch_to_block(mandel_check);
    let zx = builder.use_var(v_zx);
    let zy = builder.use_var(v_zy);
    let zx2 = builder.ins().fmul(zx, zx);
    let zy2 = builder.ins().fmul(zy, zy);
    let mag2 = builder.ins().fadd(zx2, zy2);
    let four = builder.ins().f64const(4.0);
    let escaped = builder.ins().fcmp(FloatCC::GreaterThan, mag2, four);
    builder.ins().brif(escaped, mandel_done, &[], mandel_body, &[]);

    // -- Mandel body: check iter limit, then compute step --
    builder.switch_to_block(mandel_body);
    builder.seal_block(mandel_body);
    let iter = builder.use_var(v_iter);
    let max = builder.ins().iconst(I32, max_iter as i64);
    let at_max = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, iter, max);
    let mandel_step = builder.create_block();
    builder.ins().brif(at_max, mandel_done, &[], mandel_step, &[]);

    // -- Mandel step: zx_new = zx²-zy²+cx, zy_new = 2*zx*zy+cy, iter++ --
    builder.switch_to_block(mandel_step);
    builder.seal_block(mandel_step);
    let zx = builder.use_var(v_zx);
    let zy = builder.use_var(v_zy);
    let zx2 = builder.ins().fmul(zx, zx);
    let zy2 = builder.ins().fmul(zy, zy);
    let zx2_minus_zy2 = builder.ins().fsub(zx2, zy2);

    let col = builder.use_var(v_col);
    let x_min = builder.use_var(v_x_min);
    let x_step = builder.use_var(v_x_step);
    let col_f64 = builder.ins().fcvt_from_sint(F64, col);
    let cx = builder.ins().fma(col_f64, x_step, x_min);
    let zx_new = builder.ins().fadd(zx2_minus_zy2, cx);

    let row = builder.use_var(v_row);
    let y_min = builder.use_var(v_y_min);
    let y_step = builder.use_var(v_y_step);
    let row_f64 = builder.ins().fcvt_from_sint(F64, row);
    let cy = builder.ins().fma(row_f64, y_step, y_min);

    let two = builder.ins().f64const(2.0);
    let two_zx = builder.ins().fmul(two, zx);
    let two_zx_zy = builder.ins().fmul(two_zx, zy);
    let zy_new = builder.ins().fadd(two_zx_zy, cy);

    builder.def_var(v_zx, zx_new);
    builder.def_var(v_zy, zy_new);
    let iter = builder.use_var(v_iter);
    let one = builder.ins().iconst(I32, 1);
    let iter_inc = builder.ins().iadd(iter, one);
    builder.def_var(v_iter, iter_inc);
    builder.ins().jump(mandel_check, &[]);

    // Seal mandel_check (predecessors: mandel_init, mandel_step)
    builder.seal_block(mandel_check);

    // -- Mandel done: convert iter to color, store pixel --
    builder.switch_to_block(mandel_done);
    builder.seal_block(mandel_done);

    let iter = builder.use_var(v_iter);
    let max_val = builder.ins().iconst(I32, max_iter as i64);
    let is_max = builder.ins().icmp(IntCC::Equal, iter, max_val);

    let iter_f64 = builder.ins().fcvt_from_sint(F64, iter);
    let max_f64 = builder.ins().f64const(max_iter as f64);
    let t = builder.ins().fdiv(iter_f64, max_f64);
    let one_f = builder.ins().f64const(1.0);
    let one_minus_t = builder.ins().fsub(one_f, t);
    let c255 = builder.ins().f64const(255.0);

    // r = 9.0 * (1-t) * t^3 * 255.0
    let c9 = builder.ins().f64const(9.0);
    let t2 = builder.ins().fmul(t, t);
    let t3 = builder.ins().fmul(t2, t);
    let r_f = builder.ins().fmul(c9, one_minus_t);
    let r_f = builder.ins().fmul(r_f, t3);
    let r_f = builder.ins().fmul(r_f, c255);
    let r_i = builder.ins().fcvt_to_sint(I32, r_f);
    let c255_i = builder.ins().iconst(I32, 255);
    let r_clamped = builder.ins().smin(r_i, c255_i);
    let zero_i = builder.ins().iconst(I32, 0);
    let r_clamped = builder.ins().smax(r_clamped, zero_i);

    // g = 15.0 * (1-t)^2 * t^2 * 255.0
    let c15 = builder.ins().f64const(15.0);
    let omt2 = builder.ins().fmul(one_minus_t, one_minus_t);
    let g_f = builder.ins().fmul(c15, omt2);
    let g_f = builder.ins().fmul(g_f, t2);
    let g_f = builder.ins().fmul(g_f, c255);
    let g_i = builder.ins().fcvt_to_sint(I32, g_f);
    let g_clamped = builder.ins().smin(g_i, c255_i);
    let g_clamped = builder.ins().smax(g_clamped, zero_i);

    // b = 8.5 * (1-t)^3 * t * 255.0
    let c8_5 = builder.ins().f64const(8.5);
    let omt3 = builder.ins().fmul(omt2, one_minus_t);
    let b_f = builder.ins().fmul(c8_5, omt3);
    let b_f = builder.ins().fmul(b_f, t);
    let b_f = builder.ins().fmul(b_f, c255);
    let b_i = builder.ins().fcvt_to_sint(I32, b_f);
    let b_clamped = builder.ins().smin(b_i, c255_i);
    let b_clamped = builder.ins().smax(b_clamped, zero_i);

    // color = (r << 16) | (g << 8) | b
    let sixteen = builder.ins().iconst(I32, 16);
    let eight = builder.ins().iconst(I32, 8);
    let r_shifted = builder.ins().ishl(r_clamped, sixteen);
    let g_shifted = builder.ins().ishl(g_clamped, eight);
    let color_smooth = builder.ins().bor(r_shifted, g_shifted);
    let color_smooth = builder.ins().bor(color_smooth, b_clamped);

    let black = builder.ins().iconst(I32, 0);
    let color = builder.ins().select(is_max, black, color_smooth);

    // Store: output[(row - row_start) * width + col] = color
    let row = builder.use_var(v_row);
    let row_start = builder.use_var(v_row_start);
    let width = builder.use_var(v_width);
    let col = builder.use_var(v_col);
    let row_off = builder.ins().isub(row, row_start);
    let row_off_wide = builder.ins().uextend(I64, row_off);
    let width_wide = builder.ins().uextend(I64, width);
    let col_wide = builder.ins().uextend(I64, col);
    let idx = builder.ins().imul(row_off_wide, width_wide);
    let idx = builder.ins().iadd(idx, col_wide);
    let four_bytes = builder.ins().iconst(I64, 4);
    let byte_off = builder.ins().imul(idx, four_bytes);
    let output = builder.use_var(v_output);
    let addr = builder.ins().iadd(output, byte_off);
    builder
        .ins()
        .store(cranelift_codegen::ir::MemFlags::new(), color, addr, 0);

    builder.ins().jump(inner_inc, &[]);

    // -- Inner inc: col++ --
    builder.switch_to_block(inner_inc);
    builder.seal_block(inner_inc);
    let col = builder.use_var(v_col);
    let one = builder.ins().iconst(I32, 1);
    let col_next = builder.ins().iadd(col, one);
    builder.def_var(v_col, col_next);
    builder.ins().jump(inner_check, &[]);

    // Seal inner_check (predecessors: col_init, inner_inc)
    builder.seal_block(inner_check);

    // -- Outer inc: row++ --
    builder.switch_to_block(outer_inc);
    builder.seal_block(outer_inc);
    let row = builder.use_var(v_row);
    let one = builder.ins().iconst(I32, 1);
    let row_next = builder.ins().iadd(row, one);
    builder.def_var(v_row, row_next);
    builder.ins().jump(outer_check, &[]);

    // Seal outer_check (predecessors: entry, outer_inc)
    builder.seal_block(outer_check);

    // -- Exit --
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    builder.ins().return_(&[]);

    builder.finalize();

    let mut ctx = cranelift_codegen::Context::for_function(func);
    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(func_id);
    let fn_ptr: TileKernelFn = unsafe { std::mem::transmute(code_ptr) };

    CraneliftKernel {
        _module: module,
        fn_ptr,
    }
}
