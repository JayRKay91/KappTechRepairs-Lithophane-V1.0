import numpy as np
from stl import mesh
from core.manifest_parser import LithoManifest
from core.heightmap import generate_heightmap


def build_rectangular_litho(manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15):
    """
    Constructs a watertight, manifold solid lithophane STL mesh
    including a 5mm outer frame and optional hanging eyelet tabs.
    """
    # 1. Generate core lithophane heightmap
    core_w = manifest.width_mm
    core_h = manifest.height_mm
    b_w = manifest.border_width_mm
    b_t = manifest.border_thickness_mm

    hmap, rows, cols = generate_heightmap(
        manifest.image_path,
        target_width_mm=core_w,
        target_height_mm=core_h,
        min_thickness_mm=manifest.min_thickness_mm,
        max_thickness_mm=manifest.max_thickness_mm,
        pixel_size_mm=resolution_mm
    )

    # 2. Setup grid coordinates (centered or offset by border)
    x = np.linspace(b_w, b_w + core_w, cols)
    y = np.linspace(b_w, b_w + core_h, rows)
    xx, yy = np.meshgrid(x, y)

    # 3. Create vertices for the relief surface (Z = heightmap) and flat back (Z = 0)
    top_vertices = np.column_stack((xx.ravel(), yy.ravel(), hmap.ravel()))
    bottom_vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(hmap).ravel()))

    num_pts = rows * cols
    all_vertices = np.vstack((top_vertices, bottom_vertices))

    # Helper to index grid vertices
    def idx(r, c, offset=0):
        return offset + (r * cols + c)

    faces = []

    # Grid triangulation
    r_idx, c_idx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    r_f = r_idx.ravel()
    c_f = c_idx.ravel()

    # Top textured faces (facing +Z, CCW winding)
    p1 = idx(r_f, c_f, 0)
    p2 = idx(r_f + 1, c_f, 0)
    p3 = idx(r_f + 1, c_f + 1, 0)
    p4 = idx(r_f, c_f + 1, 0)

    top_tris1 = np.column_stack((p1, p2, p3))
    top_tris2 = np.column_stack((p1, p3, p4))
    faces.extend(top_tris1.tolist())
    faces.extend(top_tris2.tolist())

    # Bottom flat faces (facing -Z, CW winding from above = CCW from bottom)
    bp1 = idx(r_f, c_f, num_pts)
    bp2 = idx(r_f + 1, c_f, num_pts)
    bp3 = idx(r_f + 1, c_f + 1, num_pts)
    bp4 = idx(r_f, c_f + 1, num_pts)

    bot_tris1 = np.column_stack((bp1, bp3, bp2))
    bot_tris2 = np.column_stack((bp1, bp4, bp3))
    faces.extend(bot_tris1.tolist())
    faces.extend(bot_tris2.tolist())

    # Side walls connecting top relief to bottom base
    # Left edge (c = 0)
    re = np.arange(rows - 1)
    faces.extend(np.column_stack((idx(re, 0, 0), idx(re, 0, num_pts), idx(re + 1, 0, 0))).tolist())
    faces.extend(np.column_stack((idx(re + 1, 0, 0), idx(re, 0, num_pts), idx(re + 1, 0, num_pts))).tolist())

    # Right edge (c = cols - 1)
    faces.extend(np.column_stack((idx(re, cols - 1, 0), idx(re + 1, cols - 1, 0), idx(re, cols - 1, num_pts))).tolist())
    faces.extend(np.column_stack((idx(re + 1, cols - 1, 0), idx(re + 1, cols - 1, num_pts), idx(re, cols - 1, num_pts))).tolist())

    # Bottom edge (r = 0)
    ce = np.arange(cols - 1)
    faces.extend(np.column_stack((idx(0, ce, 0), idx(0, ce + 1, 0), idx(0, ce, num_pts))).tolist())
    faces.extend(np.column_stack((idx(0, ce + 1, 0), idx(0, ce + 1, num_pts), idx(0, ce, num_pts))).tolist())

    # Top edge (r = rows - 1)
    faces.extend(np.column_stack((idx(rows - 1, ce, 0), idx(rows - 1, ce, num_pts), idx(rows - 1, ce + 1, 0))).tolist())
    faces.extend(np.column_stack((idx(rows - 1, ce + 1, 0), idx(rows - 1, ce, num_pts), idx(rows - 1, ce + 1, num_pts))).tolist())

    # Helper function to generate a watertight cuboid mesh
    def create_box(x0, y0, z0, x1, y1, z1):
        v = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], # 0,1,2,3 bottom
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]  # 4,5,6,7 top
        ], dtype=np.float32)
        f = [
            [0, 2, 1], [0, 3, 2], # -Z
            [4, 5, 6], [4, 6, 7], # +Z
            [0, 1, 5], [0, 5, 4], # -Y
            [2, 3, 7], [2, 7, 6], # +Y
            [0, 4, 7], [0, 7, 3], # -X
            [1, 2, 6], [1, 6, 5]  # +X
        ]
        return v, f

    all_v_list = [all_vertices]
    all_f_list = [np.array(faces, dtype=np.int32)]
    v_count = len(all_vertices)

    # 4. Generate 5mm Outer Frame Borders (Left, Right, Bottom, Top)
    total_w = core_w + 2 * b_w
    total_h = core_h + 2 * b_w

    boxes = [
        (0, 0, 0, b_w, total_h, b_t),                           # Left rail
        (total_w - b_w, 0, 0, total_w, total_h, b_t),           # Right rail
        (b_w, 0, 0, total_w - b_w, b_w, b_t),                   # Bottom rail
        (b_w, total_h - b_w, 0, total_w - b_w, total_h, b_t),   # Top rail
    ]

    # 5. Add hanging eyelet tabs if requested
    if manifest.has_hooks:
        tab_w = manifest.hook_tab_width_mm
        tab_h = 6.0
        # Two tabs near the top left and top right
        tab_left_x = b_w + (core_w * 0.2) - (tab_w / 2)
        tab_right_x = b_w + (core_w * 0.8) - (tab_w / 2)
        boxes.append((tab_left_x, total_h, 0, tab_left_x + tab_w, total_h + tab_h, b_t))
        boxes.append((tab_right_x, total_h, 0, tab_right_x + tab_w, total_h + tab_h, b_t))

    for bx0, by0, bz0, bx1, by1, bz1 in boxes:
        bv, bf = create_box(bx0, by0, bz0, bx1, by1, bz1)
        all_v_list.append(bv)
        all_f_list.append(np.array(bf, dtype=np.int32) + v_count)
        v_count += len(bv)

    final_vertices = np.vstack(all_v_list)
    final_faces = np.vstack(all_f_list)

    # 6. Save binary STL
    litho_mesh = mesh.Mesh(np.zeros(final_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(final_faces):
        for j in range(3):
            litho_mesh.vectors[i][j] = final_vertices[f[j], :]

    litho_mesh.save(output_stl_path)
    return output_stl_path