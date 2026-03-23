struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    dt: f32,
    dissipation: f32,
    buoyancy: f32,
    _pad1: u32,
};

// --- Advect pass ---
@group(0) @binding(0) var<uniform> advect_params: Params;
@group(0) @binding(1) var<storage, read>       advect_in:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> advect_out: array<vec4<f32>>;

fn idx(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = advect_params.width;
    let h = advect_params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);
    let cell = advect_in[i];

    // Semi-Lagrangian: trace backward
    let src_x = f32(x) - advect_params.dt * cell.x;
    let src_y = f32(y) - advect_params.dt * cell.y;

    // Bilinear interpolation (inlined — WGSL can't pass storage ptrs to functions)
    let cx = clamp(src_x, 0.0, f32(w) - 1.001);
    let cy = clamp(src_y, 0.0, f32(h) - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let sx = cx - floor(cx);
    let sy = cy - floor(cy);
    let v00 = advect_in[idx(x0, y0, w)];
    let v10 = advect_in[idx(x1, y0, w)];
    let v01 = advect_in[idx(x0, y1, w)];
    let v11 = advect_in[idx(x1, y1, w)];
    let top = mix(v00, v10, sx);
    let bot = mix(v01, v11, sx);
    var result = mix(top, bot, sy);

    // Buoyancy: density pushes upward (negative y direction in screen coords)
    result.y -= advect_params.buoyancy * result.z * advect_params.dt;

    // Density dissipation
    result.z *= advect_params.dissipation;

    // Open boundaries: zero velocity at edges
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        result.x = 0.0;
        result.y = 0.0;
    }

    advect_out[i] = result;
}

// --- Divergence pass ---
@group(0) @binding(0) var<uniform> div_params: Params;
@group(0) @binding(1) var<storage, read>       div_field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> div_out:   array<f32>;

@compute @workgroup_size(16, 16)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = div_params.width;
    let h = div_params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);

    // Boundary: divergence = 0
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        div_out[i] = 0.0;
        return;
    }

    let vx_r = div_field[idx(x + 1u, y, w)].x;
    let vx_l = div_field[idx(x - 1u, y, w)].x;
    let vy_d = div_field[idx(x, y + 1u, w)].y;
    let vy_u = div_field[idx(x, y - 1u, w)].y;

    div_out[i] = (vx_r - vx_l + vy_d - vy_u) * 0.5;
}

// --- Jacobi pressure solve (one iteration) ---
@group(0) @binding(0) var<uniform> jacobi_params: Params;
@group(0) @binding(1) var<storage, read>       jacobi_div:      array<f32>;
@group(0) @binding(2) var<storage, read>       jacobi_press_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> jacobi_press_out: array<f32>;

@compute @workgroup_size(16, 16)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = jacobi_params.width;
    let h = jacobi_params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);

    // Boundary: pressure = 0
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        jacobi_press_out[i] = 0.0;
        return;
    }

    let p_l = jacobi_press_in[idx(x - 1u, y, w)];
    let p_r = jacobi_press_in[idx(x + 1u, y, w)];
    let p_u = jacobi_press_in[idx(x, y - 1u, w)];
    let p_d = jacobi_press_in[idx(x, y + 1u, w)];
    let d = jacobi_div[i];

    jacobi_press_out[i] = (p_l + p_r + p_u + p_d - d) * 0.25;
}

// --- Pressure projection ---
@group(0) @binding(0) var<uniform> proj_params: Params;
@group(0) @binding(1) var<storage, read>       proj_pressure: array<f32>;
@group(0) @binding(2) var<storage, read>       proj_field_in:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> proj_field_out: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = proj_params.width;
    let h = proj_params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);
    var cell = proj_field_in[i];

    // Interior: subtract pressure gradient
    if x > 0u && x < w - 1u && y > 0u && y < h - 1u {
        let p_r = proj_pressure[idx(x + 1u, y, w)];
        let p_l = proj_pressure[idx(x - 1u, y, w)];
        let p_d = proj_pressure[idx(x, y + 1u, w)];
        let p_u = proj_pressure[idx(x, y - 1u, w)];

        cell.x -= 0.5 * (p_r - p_l);
        cell.y -= 0.5 * (p_d - p_u);
    } else {
        // Boundary: zero velocity
        cell.x = 0.0;
        cell.y = 0.0;
    }

    proj_field_out[i] = cell;
}

// --- Visualization ---
@group(0) @binding(0) var<uniform> vis_params: Params;
@group(0) @binding(1) var<storage, read>       vis_field:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vis_pixels: array<u32>;

@compute @workgroup_size(16, 16)
fn visualize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= vis_params.width || y >= vis_params.height { return; }

    let field_idx = y * vis_params.width + x;
    let pixel_idx = y * vis_params.stride + x;

    let density = vis_field[field_idx].z;
    let intensity = clamp(density, 0.0, 1.0);

    let v = u32(intensity * 255.0);
    vis_pixels[pixel_idx] = 0xFF000000u | (v << 16u) | (v << 8u) | v;
}
