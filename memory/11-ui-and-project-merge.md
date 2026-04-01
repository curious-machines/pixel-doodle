# UI Discussion Notes — 2026-03-28

## Context

With PD/PDIR removed and WGSL as the sole kernel language, the next step is adding a UI to make experimentation interactive. During discussion, a larger question emerged: should pixel-doodle and vector-flow be combined into a single project?

## UI Requirements (agreed)

### Purpose
- Interactive parameter tweaking (sliders, checkboxes from PDP variables)
- Buffer/pixel inspectors and performance overlays
- Image and video export (ffmpeg frame-by-frame, H.264)
- WGSL/PDP editing stays in VS Code (hot-reload on file change)

### Technology
- **Desktop app** using egui (eframe + egui-wgpu)
- Vector-flow used egui successfully; only egui-snarl (node graph widget) needed forking — core egui was fine
- ffmpeg pipe approach from vector-flow is directly reusable

### Layout
- Center: render viewport
- Sidebar: parameter controls, buffer inspectors
- Bottom bar: performance stats, frame timing

### Interaction Model — "Always Live"
- VS Code is the design tool (edit PDP/WGSL files)
- The app is the runtime tool (renders, runs, shows controls)
- Hot-reload on file change
- Parameter tweaks in UI are transient unless explicitly saved

### Parameter Controls
- PDP `var` declarations are the mechanism — no special "expose" syntax needed
- UI auto-generates controls from types (slider for numeric, checkbox for bool)
- Kernel args are passed through existing `run` statement:
  ```
  var max_iter: u32 = 256
  pipeline gpu {
    pixel kernel "mandelbrot.wgsl"
    run mandelbrot(max_iter: max_iter)
    display
  }
  ```
- Future: optional hints for range/step/labels (e.g., `# range: 1..2000`)

## Combining pixel-doodle and vector-flow

### What each project brings

| | pixel-doodle | vector-flow |
|---|---|---|
| Compute | WGSL shaders, GPU+CPU JIT | CPU node graph evaluation |
| Rendering | Pixel buffers via compute | Vector paths + text via wgpu render pipeline |
| Config | PDP pipeline DSL | Node graph (visual programming) |
| UI | Bare winit window | Full egui app (panels, transport, export) |
| Export | PPM screenshots | PNG + ffmpeg video |
| Unique value | GPU compute, simulations | Vector primitives, text, node composition |

### The combined vision
A creative tool where vector graphics and compute shaders share buffers. Render vectors to a buffer, process with a shader. Use a shader to generate displacement, apply to vector paths. Text as first-class alongside procedural pixel generation.

### Recommended approach: new project
Neither project is the natural host. The integration point is the shared buffer — a new abstraction neither has. A new project lets us cherry-pick the good parts without dragging exploratory baggage:

- From vector-flow: wgpu render pipeline, text system, egui app shell, export infrastructure
- From pixel-doodle: WGSL compilation, PDP config, GPU sim runner, progressive sampling

### Rough architecture
```
┌─────────────────────────────────────┐
│  egui app shell (from vector-flow)  │
│  panels, transport, export          │
├─────────────────────────────────────┤
│  Project model (evolved PDP)        │
│  - layers (vector, shader, text)    │
│  - shared buffers                   │
│  - variables / parameters / events  │
├──────────┬──────────────────────────┤
│ Vector   │  Shader engine           │
│ renderer │  (from pixel-doodle)     │
│ (from    │  - WGSL compile          │
│ v-flow)  │  - GPU dispatch          │
│ paths,   │  - CPU JIT fallback      │
│ text,    │  - sim runner            │
│ fills    │  - progressive sampling  │
├──────────┴──────────────────────────┤
│  wgpu device / shared buffers       │
└─────────────────────────────────────┘
```

## Open Questions
- Build pixel-doodle UI first then merge, or start the combined project directly?
- What is the unified project format? Evolved PDP, or something new?
- Node graph from vector-flow: keep as an alternative to PDP text, or drop it?
- How do vector layers and shader pipelines compose? (layer stack? explicit buffer routing?)
