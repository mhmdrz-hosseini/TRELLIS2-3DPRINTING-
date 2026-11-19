# TRELLIS.2 → 3D PRINTING 🖨️

**A detailed guide to Microsoft's TRELLIS.2 image-to-3D model, and how to take its output from a PBR-textured digital asset to a real, physical 3D print.**

> TRELLIS.2 generates beautiful meshes — but generative meshes are *not* print-ready by default.
> This repo documents the full pipeline: **image → TRELLIS.2 → mesh repair → scaling → STL/3MF → slicer → print.**

---

## 📑 Table of Contents

1. [What is TRELLIS.2?](#1-what-is-trellis2)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [How the Image-to-3D Pipeline Works](#3-how-the-image-to-3d-pipeline-works)
4. [Hardware Requirements](#4-hardware-requirements)
5. [Why Generative Meshes Are Not Print-Ready](#5-why-generative-meshes-are-not-print-ready)
6. [The Full 3D-Printing Pipeline](#6-the-full-3d-printing-pipeline)
7. [ComfyUI Integration](#7-comfyui-integration)
8. [Practical Example (Python)](#8-practical-example-python)
9. [TRELLIS 1 vs TRELLIS.2 Comparison](#9-trellis-1-vs-trellis2-comparison)
10. [Print Quality Tips](#10-print-quality-tips)
11. [Resources & Links](#11-resources--links)

---

## 1. What is TRELLIS.2?

**TRELLIS.2** is Microsoft's state-of-the-art large **3D generative model** (released December 2025), designed for high-fidelity **image-to-3D** generation. Given a *single image*, it produces a fully textured mesh with **PBR (Physically Based Rendering) material attributes**.

| Spec | Value |
| :--- | :--- |
| **Parameters** | 4 Billion (`TRELLIS.2-4B`) |
| **Resolution** | Up to **1536³** volumetric output |
| **Output** | Textured mesh with PBR attributes (Base Color, Metallic, Roughness, Alpha/Opacity) |
| **Topology** | Handles **open surfaces**, **non-manifold geometry**, and **internal enclosed structures** |
| **Speed** (H100) | ~3s @ 512³ · ~17s @ 1024³ · ~60s @ 1536³ |
| **License** | MIT (code & weights) — *assets are for academic/research use, **not commercial*** |
| **Weights** | [Hugging Face — microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| **Code** | [GitHub — microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| **Paper** | [arXiv 2512.14692](https://arxiv.org/abs/2512.14692) |
| **Demo** | [HF Spaces](https://huggingface.co/spaces/microsoft/TRELLIS.2) · [Project Page](https://microsoft.github.io/TRELLIS.2/) |

Unlike TRELLIS 1 (which relied on generating novel multi-view images first), TRELLIS.2 works on **native 3D latents** — it never needs an image model to invent viewpoints. That means: fewer hallucinations, cleaner geometry, and dramatically better consistency for solid objects.

---

## 2. Architecture Deep Dive

### 2.1 The three building blocks

```
Image ──▶ DiT (Diffusion Transformer) ──▶ Structured Latent ──▶ SC-VAE decoder ──▶ O-Voxel ──▶ Mesh + PBR
              (4B params, generates              (compact, sparse)      (16× compression)
               the latent grid)
```

| Component | Role |
| :--- | :--- |
| **Vanilla DiT** | A 4B-parameter Diffusion Transformer that generates the *structured latent* directly in 3D space — no multi-view image generation, no SDS optimization. |
| **SC-VAE** (Sparse Compression 3D VAE) | Encodes assets with **16× spatial downsampling** using a Sparse Residual Autoencoding scheme that compresses voxel data directly. A full 1024³ asset becomes only **~9.6K latent tokens** — this is what makes high-res generation fast. |
| **O-Voxel** | A "field-free" sparse voxel structure that stores **geometry AND appearance together**. |

### 2.2 O-Voxel: the key innovation

Classic 3D generative models decode geometry as an **iso-surface field** (e.g., Signed Distance Fields + Marching Cubes). That approach has a hard constraint: the surface must be **manifold** and **closed**. Real-world assets — clothing, foliage, thin panels, hollow shells — break that assumption.

O-Voxel breaks this with two ideas:

- **Geometry → Flexible Dual Grids** — allows **arbitrary topology with sharp edges**: open surfaces (leaves, cloth), non-manifold wire structures, and even fully **enclosed interior structures** (a box you can't see inside).
- **Appearance → full PBR attribute channels** — Base Color, Metallic, Roughness, and **Alpha (opacity)**, all stored in the voxel attributes and decoded onto the mesh via texture baking.

### 2.3 Why this matters *specifically for 3D printing*

The "arbitrary topology" freedom is a double-edged sword for printing:

- ✅ It means the model can faithfully reproduce the *look* of thin panels, lattice shells, and open cloths.
- ⚠️ But a slicer needs **watertight, manifold, solid** geometry (with a few exceptions like edge-shells and surface mode). Open surfaces from TRELLIS.2 must be **repaired/solidified** before printing — see [Section 5](#5-why-generative-meshes-are-not-print-ready).

---

## 3. How the Image-to-3D Pipeline Works

### Step-by-step

1. **Input image** — one photo/render of the object (clean background, good lighting recommended).
2. **Latent generation** — the DiT denoises a sparse structured latent grid conditioned on the image.
3. **VAE decoding** — the SC-VAE decodes the compact latents back into a dense **O-Voxel** grid at the chosen resolution (512³ / 1024³ / 1536³).
4. **Mesh extraction** — O-Voxel → textured mesh. This is *instant*: **<100 ms on CUDA**, with no rendering-based optimization or fitting loop. The reverse direction (textured mesh → O-Voxel) takes **<10 s on a single CPU**, which is what makes training data preparation feasible.
5. **Mesh simplification** — cap faces for export (nvdiffrast limit) or set a decimation target (~1M faces default in the examples).
6. **Export** — GLB with baked PBR textures.

### Generation speed (NVIDIA H100)

| Resolution | Total | Shape + Material |
| :--- | :--- | :--- |
| 512³ | **~3 s** | 2s + 1s |
| 1024³ | **~17 s** | 10s + 7s |
| 1536³ | **~60 s** | 35s + 25s |

On consumer GPUs (e.g., RTX 3060/4090) expect roughly 3–10× slower per stage, and 1536³ may need memory tricks (see `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

---

## 4. Hardware Requirements

From the official repo:

- **OS:** Linux tested (Windows works via WSL2)
- **GPU:** **NVIDIA with ≥ 24 GB VRAM** — verified on A100 / H100. Consumer cards with 12–16 GB can run 512³–1024³ with `expandable_segments` and lower resolution; 1536³ is demanding.
- **CUDA:** Toolkit 12.4 recommended (needed to compile custom kernels)
- **Python:** ≥ 3.8, Conda recommended
- **Attention backend:** `flash-attn` by default; `xformers` fallback for older GPUs (set `ATTN_BACKEND=xformers`)

### Install (official)

```sh
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

> ⚠️ On Windows, run this inside **WSL2** — the custom CUDA extensions (`cumesh`, `o-voxel`, `flexgemm`) don't build natively on Windows.

---

## 5. Why Generative Meshes Are Not Print-Ready

This is the heart of "TRELLIS.2 for 3D printing." The model's strengths create exactly the four problems a slicer hates:

| TRELLIS.2 feature | Why it's a printing problem | Fix |
| :--- | :--- | :--- |
| **Open surfaces** (cloth, leaves, panels) | Slicer sees a zero-thickness sheet → holes in the print, or silent auto-repair garbage | Thicken with a solidify modifier (Blender) or repair tools |
| **Non-manifold geometry** | Some slicers (Cura) refuse or "heal" it unpredictably | Mesh repair: PrusaSlicer/OrcaSlicer auto-repair, Blender 3D-Print Toolbox, Meshmixer, Netfabb |
| **Alpha channel** (transparency) | Printed as opaque geometry where alpha>0; thin translucent wisps break | Decide print intent: solidify the alpha areas or delete them |
| **Unit scale** (`AABB = [-0.5, 0.5]`) | Mesh is ~1 unit ≈ meaningless for printers | Scale to mm in Blender/slicer (e.g., ×100 → 100 mm object) |
| **High polygon count** (up to 16M internally, ~1M after decimation) | Slicers choke on huge meshes | Decimate to 100–300k faces; slicing precision rarely needs more |

**Rule of thumb:** *manifold + watertight + ≥2 mm wall thickness + real-world scale = printable.*

---

## 6. The Full 3D-Printing Pipeline

```
┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐
│ Source     │──▶│ TRELLIS.2 │──▶│ GLB export   │──▶│ Repair &   │──▶│ STL / 3MF│──▶│ Slicer │──▶ 🖨️
│ image      │   │ (512³–1536³)│  │ (geometry +  │   │ scale      │   │ export   │   │ (Orca/ │
│            │   │           │   │  PBR texture)│   │ + solidify │   │          │   │ Cura/  │
│            │   │           │   │              │   │            │   │          │   │ Prusa) │
└────────────┘   └───────────┘   └──────────────┘   └────────────┘   └──────────┘   └────────┘
```

### Step 1 — Generate

- Choose resolution by use case: **512³** for quick drafts, **1024³** for good detail, **1536³** for max fidelity (small decorative parts).
- Decimation target in `to_glb(...)`: `decimation_target=1_000_000` is the default — lower it to 100k–300k for smoother slicing.

### Step 2 — Export geometry only

For printing, **textures don't matter** (unless multi-material color printing). Export **STL** (geometry-only, universally supported) or **3MF** (keeps colors + units, preferred for Bambu/Prusa multi-material).

### Step 3 — Repair & inspect (5 minutes, mandatory)

Pick one:

| Tool | Best for |
| :--- | :--- |
| **Blender 3D-Print Toolbox** (built-in addon) | Checking manifold/watertight, hole filling, solidify, precise scale |
| **PrusaSlicer / OrcaSlicer** built-in repair | Fast "just make it work" auto-repair on import |
| **Meshmixer** | Solidifying open sheets, thickening, hollowing |
| **Netfabb / Windows 3D Print repair** | One-click repair of STL |

Checks to run:
- ✅ Watertight (no open edges)
- ✅ Manifold (consistent normals, no self-intersections)
- ✅ Wall thickness ≥ 2× nozzle width (≥ 0.8 mm for a 0.4 mm nozzle; 2 mm+ for handling strength)

### Step 4 — Scale & orient

- TRELLIS.2's output occupies a unit cube (`aabb = [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]`) — multiply to target size in mm.
- Orient **flat side down** on the build plate; keep overhangs ≤ 45° where possible.

### Step 5 — Slice

- **Material:** PLA for decorative; PETG/ABS/ASA for functional; resin printer for high detail (1024³+ assets shine on resin).
- **Hollow** large solid meshes (Blender "Hollow" in 3D-Print Toolbox, or slicer infill) to save filament.
- **Supports:** enable for overhangs; tree supports are gentler on organic shapes (which is exactly what TRELLIS.2 tends to produce).
- **Layer height:** 0.12–0.2 mm typical; resin 0.025–0.05 mm for miniatures.

### Step 6 — Print & post-process

Support removal → sanding (layer lines smooth out well on organic shapes) → optional primer+paint; the PBR textures are a good *paint reference* even though they don't transfer to FDM.

---

## 7. ComfyUI Integration

The best news for ComfyUI users: **TRELLIS.2 became a native ComfyUI node in August 2026** (alongside Pixal3D) — no custom node installation required for the core pipeline.

- **Native nodes** include: TRELLIS.2 image-to-3D (shape + texture stages), MoGe depth conditioning, mesh post-processing (re-implemented as pure nodes), and an "Export Mesh" node compatible with the "Preview 3D" node.
- Official tutorial: [docs.comfy.org — TRELLIS.2 Image to 3D](https://docs.comfy.org/tutorials/3d/trellis2)
- A community wrapper custom node also exists with extras (Pixal3D MultiView support, "Smooth" nodes).
- Community wrapper: [GitHub — ComfyUI-TRELLIS.2 wrapper](https://github.com/Za0Bleu/comfyui-TRELLIS.2-wrapper)

### ComfyUI + printing workflow

```
ComfyUI (text-to-image) ──▶ TRELLIS.2 node (image-to-3D) ──▶ Export Mesh (GLB/OBJ)
                                                              │
                                             Blender: repair + solidify + scale to mm
                                                              │
                                              OrcaSlicer / Bambu Studio ──▶ 🖨️
```

This gives you a fully node-based pipeline: prompt → product shot → 3D asset → print file.

---

## 8. Practical Example (Python)

See [`example_image_to_stl.py`](example_image_to_stl.py) in this repo — an end-to-end script that runs the official TRELLIS.2 pipeline and then converts the result into a **print-ready STL/3MF** (watertight check, normal fixing, real-world mm scaling) using `trimesh`.

Quick skeleton (from the official minimal example + print conversion):

```python
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import trimesh
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# 1. Generate
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 2. Export GLB (official postprocessing)
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
    coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=200_000,   # ↓ from 1M — better for slicing
    texture_size=4096, remesh=True, remesh_band=1, remesh_project=0,
)
glb.export("sample.glb", extension_webp=True)

# 3. Convert to print-ready STL (see full script in this repo)
m = trimesh.load("sample.glb")
m = m.dump(concatenate=True) if hasattr(m, "dump") else m
trimesh.repair.fix_normals(m)
trimesh.repair.fill_holes(m)
print(f"watertight={m.is_watertight}  faces={len(m.faces)}")
m.apply_scale(100.0)          # unit cube → 100 mm object
m.export("print.stl")
```

---

## 9. TRELLIS 1 vs TRELLIS.2 Comparison

| | TRELLIS (1) | TRELLIS.2 |
| :--- | :--- | :--- |
| Release | Dec 2024 | Dec 2025 |
| Latent space | Structured latents + 2D image prior for views | **Native 3D** structured latents (no multi-view pass) |
| Geometry decode | SDF iso-surface (marching cubes) | **O-Voxel + Flexible Dual Grids** |
| Topology | Mostly closed manifolds | **Open surfaces, non-manifold, enclosed interiors** |
| Materials | Basic color textures | **Full PBR**: Base Color, Metallic, Roughness, **Alpha** |
| Resolution | ~512³–1024³ equivalent | **512³–1536³** |
| Size | ~2×1.2B | **4B** |
| Relighting | Limited | Photorealistic PBR relighting |

**For printing:** TRELLIS.2's geometry quality (sharp edges, consistent solids) is a major upgrade — but its support for open/non-manifold surfaces means the **repair step is now mandatory**, while TRELLIS 1's SDF output was often accidentally watertight.

---

## 10. Print Quality Tips

1. **Prompt for solidity.** Objects that look solid in the image (statues, tools, housings) generate solid. Avoid images of sheer cloth, wires, or thin lace for FDM prints — they generate as open sheets.
2. **Clean input images win.** Plain background, even lighting, object fully in frame, no cropping at edges.
3. **1024³ is the sweet spot** for FDM detail; 1536³ mostly helps small/resin parts.
4. **Decimate before slicing.** 150k–300k faces is plenty; 1M+ faces slows slicers 10× with no visual gain at 0.2 mm layers.
5. **Hollow + drain holes.** For resin printing, hollow the mesh in the slicer and add ≥ 3 mm drain holes — resin uncured inside a hollow shell will crack it.
6. **Check at 10× zoom in Blender.** Inspect the mesh in wireframe before sending it to the slicer; catching an open sheet early saves a 12-hour failed print.
7. **Verify scale in the slicer.** Generative models don't know real-world size — always set dimensions explicitly (e.g., X = 80 mm) before slicing.
8. **License check.** TRELLIS.2 outputs are for **academic/research use** — don't sell printed derivatives without confirming rights.

---

## 11. Resources & Links

- 📄 **Paper:** [arXiv 2512.14692 — Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692)
- 💻 **Code:** [github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT)
- 🤗 **Weights:** [huggingface.co/microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
- 🌐 **Project page:** [microsoft.github.io/TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
- 🖥️ **Free demo:** [HF Spaces — microsoft/TRELLIS.2](https://huggingface.co/spaces/microsoft/TRELLIS.2)
- 🧩 **ComfyUI:** [Official TRELLIS.2 tutorial](https://docs.comfy.org/tutorials/3d/trellis2) · [TRELLIS.2 native in ComfyUI (blog)](https://blog.comfy.org)
- 🧰 **Repair tools:** [Blender 3D-Print Toolbox](https://docs.blender.org/manual/en/latest/addons/mesh/3d_print_toolbox.html) · Meshmixer · OrcaSlicer
- 📌 **This repo:** `example_image_to_stl.py` — end-to-end generation → print-ready STL script

---

*Last updated: 2026-11-19 · Maintained by [@mhmdrz-hosseini](https://github.com/mhmdrz-hosseini)*
