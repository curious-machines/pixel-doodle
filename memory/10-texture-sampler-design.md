# Texture Sampler Configuration Design

## Status: Proposed (not yet implemented)

## Current State

- `tex_sample` hardcodes bilinear + repeat
- `tex_load` hardcodes repeat
- No way to control filter or address mode from kernel or PDP

## Proposed Design

Sampler config lives on the PDP texture declaration. Kernels reference textures without specifying modes.

### PDP syntax

```
texture img = file("photo.png")                                    # defaults: bilinear, repeat
texture img = file("photo.png"), filter: nearest                   # nearest, repeat
texture img = file("photo.png"), address: clamp                    # bilinear, clamp
texture img = file("photo.png"), filter: nearest, address: clamp   # both explicit
```

### Kernel code (unchanged)

```
let rgba = tex_sample(img, u, v);   // filter + address from PDP config
let rgba = tex_load(img, x, y);     // address from PDP config
```

### Multiple sampling modes for same image

Declare twice with different names:

```
texture img_smooth = file("photo.png"), filter: bilinear, address: clamp
texture img_pixel  = file("photo.png"), filter: nearest, address: repeat
```

### Rationale

- Follows WGSL model: sampler is a host-side concern, not a shader concern
- Kernel code stays simple — just `tex_sample` and `tex_load`
- PDP owns configuration, consistent with how buffers/settings work
- No new syntax needed in PD or PDIR languages
- If same image needs different modes, declare it twice (the image data can be shared internally)

### Filter modes

- `bilinear` (default) — linear interpolation between 4 nearest texels
- `nearest` — nearest-neighbor, no interpolation

### Address modes

- `repeat` (default) — wraps coordinates modulo texture dimensions
- `clamp` — clamps to edge texels

### Future considerations

- `mirror` address mode
- Mipmap / anisotropic filtering
- Border color for out-of-bounds
