"""
TRELLIS.2 → Print-ready STL/3MF
===============================
End-to-end script: single image  →  TRELLIS.2-4B  →  GLB  →  repaired STL (and 3MF).

Usage:
    python example_image_to_stl.py --image assets/example_image/T.png --size-mm 100

Requirements:
    - TRELLIS.2 installed per the official repo (Linux / WSL2, conda env `trellis2`):
      https://github.com/microsoft/TRELLIS.2
    - pip install trimesh   # pure-python, no build step
"""

import os

# Keep these BEFORE importing torch/cv2 (official TRELLIS.2 recommendation)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse

import trimesh

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def generate_glb(image_path: str, out_glb: str, decimation: int = 200_000):
    """Run the official TRELLIS.2 image-to-3D pipeline and export a GLB."""
    from PIL import Image

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    image = Image.open(image_path)
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)  # cap faces (nvdiffrast limit)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  # unit cube output
        decimation_target=decimation,  # ↓ from 1M — lighter for slicers
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(out_glb, extension_webp=True)
    print(f"[+] GLB written: {out_glb}")


def make_printable(glb_path: str, out_stl: str, out_3mf: str, size_mm: float):
    """Load the GLB, repair the mesh, scale to real-world mm, export STL/3MF."""
    scene = trimesh.load(glb_path)
    mesh = scene.dump(concatenate=True) if hasattr(scene, "dump") else scene

    # --- Repair pass: fix normals, close small holes ---
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)

    # --- Printability report (the checks that matter) ---
    faces = len(mesh.faces)
    watertight = mesh.is_watertight
    winding = mesh.is_winding_consistent
    print(f"[i] faces={faces:,}  watertight={watertight}  consistent_winding={winding}")

    if not watertight:
        print("[!] NOT watertight — open sheets detected (common for cloth/foliage).")
        print("    Open it in Blender (3D-Print Toolbox) or Meshmixer and solidify before printing.")

    # --- Scale: TRELLIS.2 output spans a unit cube [-0.5, 0.5] → make it size_mm ---
    dims = mesh.extents  # (dx, dy, dz) in model units ≈ 1.0
    scale = size_mm / max(dims)
    mesh.apply_scale(scale)
    print(f"[i] scaled ×{scale:.2f} → {mesh.extents.round(1)} mm")

    # --- Export geometry-only STL (universal) and 3MF (units-aware) ---
    mesh.export(out_stl)
    mesh.export(out_3mf)
    print(f"[+] Print-ready: {out_stl}")
    print(f"[+] Print-ready: {out_3mf}")
    print("[i] Next: import into OrcaSlicer / Bambu Studio / PrusaSlicer, verify size, slice.")


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image → print-ready STL")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--out", default="trellis2_output", help="output prefix/dir")
    ap.add_argument("--size-mm", type=float, default=100.0, help="largest dimension in mm")
    ap.add_argument("--decimation", type=int, default=200_000, help="GLB face target")
    ap.add_argument("--skip-generate", action="store_true", help="reuse existing .glb")
    args = ap.parse_args()

    glb_path = f"{args.out}.glb"
    if not args.skip_generate:
        generate_glb(args.image, glb_path, decimation=args.decimation)

    make_printable(
        glb_path,
        out_stl=f"{args.out}_print.stl",
        out_3mf=f"{args.out}_print.3mf",
        size_mm=args.size_mm,
    )


if __name__ == "__main__":
    main()
