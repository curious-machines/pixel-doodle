use inkwell::context::Context;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::FunctionValue;
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};

use crate::jit::{CompiledKernel, JitBackend, TileKernelFn};
use crate::kernel_ir::KernelIr;

pub struct LlvmBackend;

struct LlvmKernel {
    // Leaked context + execution engine keep the compiled code alive.
    // We compile once per run, so leaking a few KB is fine.
    _engine: inkwell::execution_engine::ExecutionEngine<'static>,
    fn_ptr: TileKernelFn,
}

// Safety: the ExecutionEngine owns the compiled code and keeps it alive.
unsafe impl Send for LlvmKernel {}
unsafe impl Sync for LlvmKernel {}

impl CompiledKernel for LlvmKernel {
    fn function_ptr(&self) -> TileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for LlvmBackend {
    fn compile(&self, ir: &KernelIr) -> Box<dyn CompiledKernel> {
        match ir {
            KernelIr::Mandelbrot { max_iter } => Box::new(compile_mandelbrot(*max_iter)),
        }
    }
}

fn compile_mandelbrot(max_iter: u32) -> LlvmKernel {
    // Leak the context so we get 'static lifetime
    let context: &'static Context = Box::leak(Box::new(Context::create()));

    let module = context.create_module("mandelbrot");

    // Set up target
    Target::initialize_native(&InitializationConfig::default())
        .expect("failed to initialize native target");
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).unwrap();
    let machine = target
        .create_target_machine(
            &triple,
            "native",
            "",
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .unwrap();
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&triple);

    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // fn(output: ptr, width: i32, height: i32, x_min: f64, y_min: f64,
    //    x_step: f64, y_step: f64, row_start: i32, row_end: i32) -> void
    let fn_type = void_type.fn_type(
        &[
            ptr_type.into(),   // output
            i32_type.into(),   // width
            i32_type.into(),   // height
            f64_type.into(),   // x_min
            f64_type.into(),   // y_min
            f64_type.into(),   // x_step
            f64_type.into(),   // y_step
            i32_type.into(),   // row_start
            i32_type.into(),   // row_end
        ],
        false,
    );

    let function = module.add_function("mandelbrot_tile", fn_type, None);
    build_mandelbrot_body(context, &module, function, max_iter);

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    let fn_ptr = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut u32, u32, u32, f64, f64, f64, f64, u32, u32)>(
                "mandelbrot_tile",
            )
            .unwrap()
            .as_raw()
    };
    let fn_ptr: TileKernelFn = unsafe { std::mem::transmute(fn_ptr) };

    LlvmKernel {
        _engine: engine,
        fn_ptr,
    }
}

fn build_mandelbrot_body(
    context: &'static Context,
    _module: &inkwell::module::Module<'static>,
    function: FunctionValue<'static>,
    max_iter: u32,
) {
    let builder = context.create_builder();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let f64_type = context.f64_type();

    let entry = context.append_basic_block(function, "entry");
    let outer_check = context.append_basic_block(function, "outer_check");
    let inner_pre = context.append_basic_block(function, "inner_pre");
    let inner_check = context.append_basic_block(function, "inner_check");
    let mandel_pre = context.append_basic_block(function, "mandel_pre");
    let mandel_check = context.append_basic_block(function, "mandel_check");
    let mandel_body = context.append_basic_block(function, "mandel_body");
    let mandel_done = context.append_basic_block(function, "mandel_done");
    let inner_inc = context.append_basic_block(function, "inner_inc");
    let outer_inc = context.append_basic_block(function, "outer_inc");
    let exit = context.append_basic_block(function, "exit");

    // -- Entry --
    builder.position_at_end(entry);
    let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
    let p_width = function.get_nth_param(1).unwrap().into_int_value();
    let p_x_min = function.get_nth_param(3).unwrap().into_float_value();
    let p_y_min = function.get_nth_param(4).unwrap().into_float_value();
    let p_x_step = function.get_nth_param(5).unwrap().into_float_value();
    let p_y_step = function.get_nth_param(6).unwrap().into_float_value();
    let p_row_start = function.get_nth_param(7).unwrap().into_int_value();
    let p_row_end = function.get_nth_param(8).unwrap().into_int_value();

    let row_ptr = builder.build_alloca(i32_type, "row_ptr").unwrap();
    let col_ptr = builder.build_alloca(i32_type, "col_ptr").unwrap();
    let zx_ptr = builder.build_alloca(f64_type, "zx_ptr").unwrap();
    let zy_ptr = builder.build_alloca(f64_type, "zy_ptr").unwrap();
    let iter_ptr = builder.build_alloca(i32_type, "iter_ptr").unwrap();

    builder.build_store(row_ptr, p_row_start).unwrap();
    builder.build_unconditional_branch(outer_check).unwrap();

    // -- Outer check: row < row_end --
    builder.position_at_end(outer_check);
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::SLT, row, p_row_end, "row_lt").unwrap();
    builder.build_conditional_branch(cmp, inner_pre, exit).unwrap();

    // -- Inner pre: col = 0 --
    builder.position_at_end(inner_pre);
    builder.build_store(col_ptr, i32_type.const_zero()).unwrap();
    builder.build_unconditional_branch(inner_check).unwrap();

    // -- Inner check: col < width --
    builder.position_at_end(inner_check);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::SLT, col, p_width, "col_lt").unwrap();
    builder.build_conditional_branch(cmp, mandel_pre, outer_inc).unwrap();

    // -- Mandel pre: init zx=0, zy=0, iter=0 --
    builder.position_at_end(mandel_pre);
    builder.build_store(zx_ptr, f64_type.const_float(0.0)).unwrap();
    builder.build_store(zy_ptr, f64_type.const_float(0.0)).unwrap();
    builder.build_store(iter_ptr, i32_type.const_zero()).unwrap();
    builder.build_unconditional_branch(mandel_check).unwrap();

    // -- Mandel check: zx²+zy² <= 4.0 && iter < max_iter --
    builder.position_at_end(mandel_check);
    let zx = builder.build_load(f64_type, zx_ptr, "zx").unwrap().into_float_value();
    let zy = builder.build_load(f64_type, zy_ptr, "zy").unwrap().into_float_value();
    let iter = builder.build_load(i32_type, iter_ptr, "iter").unwrap().into_int_value();

    let zx2 = builder.build_float_mul(zx, zx, "zx2").unwrap();
    let zy2 = builder.build_float_mul(zy, zy, "zy2").unwrap();
    let mag2 = builder.build_float_add(zx2, zy2, "mag2").unwrap();
    let four = f64_type.const_float(4.0);
    let not_escaped = builder.build_float_compare(FloatPredicate::OLE, mag2, four, "not_escaped").unwrap();
    let max_val = i32_type.const_int(max_iter as u64, false);
    let not_maxed = builder.build_int_compare(IntPredicate::SLT, iter, max_val, "not_maxed").unwrap();
    let cont = builder.build_and(not_escaped, not_maxed, "continue").unwrap();
    builder.build_conditional_branch(cont, mandel_body, mandel_done).unwrap();

    // -- Mandel body: step --
    builder.position_at_end(mandel_body);
    let zx = builder.build_load(f64_type, zx_ptr, "zx").unwrap().into_float_value();
    let zy = builder.build_load(f64_type, zy_ptr, "zy").unwrap().into_float_value();
    let zx2 = builder.build_float_mul(zx, zx, "zx2").unwrap();
    let zy2 = builder.build_float_mul(zy, zy, "zy2").unwrap();
    let zx2_sub_zy2 = builder.build_float_sub(zx2, zy2, "zx2_sub_zy2").unwrap();

    // cx = x_min + col * x_step
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let col_f = builder.build_signed_int_to_float(col, f64_type, "col_f").unwrap();
    let col_step = builder.build_float_mul(col_f, p_x_step, "col_step").unwrap();
    let cx = builder.build_float_add(p_x_min, col_step, "cx").unwrap();

    let zx_new = builder.build_float_add(zx2_sub_zy2, cx, "zx_new").unwrap();

    // cy = y_min + row * y_step
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let row_f = builder.build_signed_int_to_float(row, f64_type, "row_f").unwrap();
    let row_step = builder.build_float_mul(row_f, p_y_step, "row_step").unwrap();
    let cy = builder.build_float_add(p_y_min, row_step, "cy").unwrap();

    let two = f64_type.const_float(2.0);
    let two_zx = builder.build_float_mul(two, zx, "two_zx").unwrap();
    let two_zx_zy = builder.build_float_mul(two_zx, zy, "two_zx_zy").unwrap();
    let zy_new = builder.build_float_add(two_zx_zy, cy, "zy_new").unwrap();

    builder.build_store(zx_ptr, zx_new).unwrap();
    builder.build_store(zy_ptr, zy_new).unwrap();
    let iter = builder.build_load(i32_type, iter_ptr, "iter").unwrap().into_int_value();
    let iter_inc = builder.build_int_add(iter, i32_type.const_int(1, false), "iter_inc").unwrap();
    builder.build_store(iter_ptr, iter_inc).unwrap();
    builder.build_unconditional_branch(mandel_check).unwrap();

    // -- Mandel done: color + store --
    builder.position_at_end(mandel_done);
    let iter = builder.build_load(i32_type, iter_ptr, "iter").unwrap().into_int_value();
    let max_val = i32_type.const_int(max_iter as u64, false);
    let is_max = builder.build_int_compare(IntPredicate::EQ, iter, max_val, "is_max").unwrap();

    // Smooth color
    let iter_f = builder.build_signed_int_to_float(iter, f64_type, "iter_f").unwrap();
    let max_f = f64_type.const_float(max_iter as f64);
    let t = builder.build_float_div(iter_f, max_f, "t").unwrap();
    let one = f64_type.const_float(1.0);
    let omt = builder.build_float_sub(one, t, "omt").unwrap();
    let c255 = f64_type.const_float(255.0);

    // r = 9.0 * (1-t) * t^3 * 255
    let c9 = f64_type.const_float(9.0);
    let t2 = builder.build_float_mul(t, t, "t2").unwrap();
    let t3 = builder.build_float_mul(t2, t, "t3").unwrap();
    let r = builder.build_float_mul(c9, omt, "r1").unwrap();
    let r = builder.build_float_mul(r, t3, "r2").unwrap();
    let r = builder.build_float_mul(r, c255, "r3").unwrap();
    let r_i = builder.build_float_to_signed_int(r, i32_type, "r_i").unwrap();

    // g = 15.0 * (1-t)^2 * t^2 * 255
    let c15 = f64_type.const_float(15.0);
    let omt2 = builder.build_float_mul(omt, omt, "omt2").unwrap();
    let g = builder.build_float_mul(c15, omt2, "g1").unwrap();
    let g = builder.build_float_mul(g, t2, "g2").unwrap();
    let g = builder.build_float_mul(g, c255, "g3").unwrap();
    let g_i = builder.build_float_to_signed_int(g, i32_type, "g_i").unwrap();

    // b = 8.5 * (1-t)^3 * t * 255
    let c8_5 = f64_type.const_float(8.5);
    let omt3 = builder.build_float_mul(omt2, omt, "omt3").unwrap();
    let b = builder.build_float_mul(c8_5, omt3, "b1").unwrap();
    let b = builder.build_float_mul(b, t, "b2").unwrap();
    let b = builder.build_float_mul(b, c255, "b3").unwrap();
    let b_i = builder.build_float_to_signed_int(b, i32_type, "b_i").unwrap();

    // Clamp to [0, 255]
    let c0 = i32_type.const_zero();
    let c255_i = i32_type.const_int(255, false);

    let r_cl = clamp_i32(&builder, r_i, c0, c255_i);
    let g_cl = clamp_i32(&builder, g_i, c0, c255_i);
    let b_cl = clamp_i32(&builder, b_i, c0, c255_i);

    // color = (r << 16) | (g << 8) | b
    let r_sh = builder.build_left_shift(r_cl, i32_type.const_int(16, false), "r_sh").unwrap();
    let g_sh = builder.build_left_shift(g_cl, i32_type.const_int(8, false), "g_sh").unwrap();
    let color = builder.build_or(r_sh, g_sh, "rg").unwrap();
    let color = builder.build_or(color, b_cl, "rgb").unwrap();

    let black = i32_type.const_zero();
    let final_color = builder.build_select::<inkwell::values::IntValue, inkwell::values::IntValue>(is_max, black.into(), color.into(), "final_color").unwrap().into_int_value();

    // Store: output[(row - row_start) * width + col] = color
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let row_off = builder.build_int_sub(row, p_row_start, "row_off").unwrap();
    let row_off64 = builder.build_int_z_extend(row_off, i64_type, "row_off64").unwrap();
    let width64 = builder.build_int_z_extend(p_width, i64_type, "width64").unwrap();
    let col64 = builder.build_int_z_extend(col, i64_type, "col64").unwrap();
    let idx = builder.build_int_mul(row_off64, width64, "idx1").unwrap();
    let idx = builder.build_int_add(idx, col64, "idx2").unwrap();

    let ptr = unsafe {
        builder.build_gep(i32_type, p_output, &[idx], "ptr").unwrap()
    };
    builder.build_store(ptr, final_color).unwrap();

    builder.build_unconditional_branch(inner_inc).unwrap();

    // -- Inner inc --
    builder.position_at_end(inner_inc);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let col_next = builder.build_int_add(col, i32_type.const_int(1, false), "col_next").unwrap();
    builder.build_store(col_ptr, col_next).unwrap();
    builder.build_unconditional_branch(inner_check).unwrap();

    // -- Outer inc --
    builder.position_at_end(outer_inc);
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let row_next = builder.build_int_add(row, i32_type.const_int(1, false), "row_next").unwrap();
    builder.build_store(row_ptr, row_next).unwrap();
    builder.build_unconditional_branch(outer_check).unwrap();

    // -- Exit --
    builder.position_at_end(exit);
    builder.build_return(None).unwrap();
}

fn clamp_i32<'ctx>(
    builder: &inkwell::builder::Builder<'ctx>,
    val: inkwell::values::IntValue<'ctx>,
    min: inkwell::values::IntValue<'ctx>,
    max: inkwell::values::IntValue<'ctx>,
) -> inkwell::values::IntValue<'ctx> {
    let cmp_lo = builder.build_int_compare(IntPredicate::SLT, val, min, "cmp_lo").unwrap();
    let v1 = builder.build_select::<inkwell::values::IntValue, inkwell::values::IntValue>(cmp_lo, min.into(), val.into(), "clamp_lo").unwrap().into_int_value();
    let cmp_hi = builder.build_int_compare(IntPredicate::SGT, v1, max, "cmp_hi").unwrap();
    builder.build_select::<inkwell::values::IntValue, inkwell::values::IntValue>(cmp_hi, max.into(), v1.into(), "clamp_hi").unwrap().into_int_value()
}
