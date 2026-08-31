import numpy as np
from stl import mesh
from core.manifest_parser import LithoManifest
from core.heightmap import generate_heightmap


def build_lithophane_stl(manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15):
    """Router that handles rectangle or circular lithophanes from the web manifest."""
    if manifest.shape == "circle":
        return build_circular_litho(manifest, output_stl_path, resolution_mm)
    else:
        return build_rectangular_litho(manifest, output_stl_path, resolution_mm)


def _create_annular_hook_tab(center_x: float, center_y: float, hole_dia: float, tab_depth: float, num_steps: int = 32):
    """Generates a rounded eyelet tab with a center hole."""
    inner_r = hole_dia / 2.0
    outer_r = inner_r + 2.5  # 2.5mm solid wall around hole

    angles = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
    r_vals = np.linspace(inner_r, outer_r, 3)
    t_r, t_th = np.meshgrid(r_vals, angles)

    hx = center_x + (t_r * np.cos(t_th))
    hy = center_y + (t_r * np.sin(t_th))

    top_v = np.column_stack((hx.ravel(), hy.ravel(), np.full(hx.size, tab_depth)))
    bot_v = np.column_stack((hx.ravel(), hy.ravel(), np.zeros(hx.size)))
    num_pts = top_v.shape[0]

    all_v = np.vstack((top_v, bot_v))
    faces = []

    for i in range(num_steps):
        i_next = (i + 1) % num_steps
        for j in range(2):
            p1 = i * 3 + j
            p2 = i * 3 + (j + 1)
            p3 = i_next * 3 + (j + 1)
            p4 = i_next * 3 + j

            # Top faces
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

            # Bottom faces
            bp1 = p1 + num_pts
            bp2 = p2 + num_pts
            bp3 = p3 + num_pts
            bp4 = p4 + num_pts
            faces.append([bp1, bp3, bp2])
            faces.append([bp1, bp4, bp3])

        # Inner hole wall
        in_t1 = i * 3
        in_t2 = i_next * 3
        in_b1 = in_t1 + num_pts
        in_b2 = in_t2 + num_pts
        faces.append([in_t1, in_t2, in_b1])
        faces.append([in_t2, in_b2, in_b1])

        # Outer perimeter wall
        out_t1 = i * 3 + 2
        out_t2 = i_next * 3 + 2
        out_b1 = out_t1 + num_pts
        out_b2 = out_t2 + num_pts
        faces.append([out_t1, out_b1, out_t2])
        faces.append([out_t2, out_b1, out_b2])

    return all_v, np.array(faces, dtype=np.int32)


def build_rectangular_litho(manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15):
    """Builds a solid rectangular lithophane with borders and optional hooks."""
    core_w = manifest.width_mm
    core_h = manifest.height_mm
    b_w = manifest.border_width_mm
    b_d = manifest.border_depth_mm

    hmap, rows, cols = generate_heightmap(
        manifest.source_image_path,
        target_width_mm=core_w,
        target_height_mm=core_h,
        min_thickness_mm=manifest.min_thickness_mm,
        max_thickness_mm=manifest.max_thickness_mm,
        pixel_size_mm=resolution_mm
    )

    x = np.linspace(b_w, b_w + core_w, cols)
    y = np.linspace(b_w, b_w + core_h, rows)
    xx, yy = np.meshgrid(x, y)

    top_v = np.column_stack((xx.ravel(), yy.ravel(), hmap.ravel()))
    bot_v = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(hmap).ravel()))
    num_pts = rows * cols
    all_vertices = np.vstack((top_v, bot_v))

    def idx(r, c, offset=0):
        return offset + (r * cols + c)

    r_idx, c_idx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing='ij')
    rf = r_idx.ravel()
    cf = c_idx.ravel()

    # Top & Bottom faces
    t1 = np.column_stack((idx(rf, cf), idx(rf + 1, cf), idx(rf + 1, cf + 1)))
    t2 = np.column_stack((idx(rf, cf), idx(rf + 1, cf + 1), idx(rf, cf + 1)))
    b1 = np.column_stack((idx(rf, cf, num_pts), idx(rf + 1, cf + 1, num_pts), idx(rf + 1, cf, num_pts)))
    b2 = np.column_stack((idx(rf, cf, num_pts), idx(rf, cf + 1, num_pts), idx(rf + 1, cf + 1, num_pts)))

    # Wall edges
    re = np.arange(rows - 1)
    ce = np.arange(cols - 1)
    w_l1 = np.column_stack((idx(re, 0), idx(re, 0, num_pts), idx(re + 1, 0)))
    w_l2 = np.column_stack((idx(re + 1, 0), idx(re, 0, num_pts), idx(re + 1, 0, num_pts)))
    w_r1 = np.column_stack((idx(re, cols - 1), idx(re + 1, cols - 1), idx(re, cols - 1, num_pts)))
    w_r2 = np.column_stack((idx(re + 1, cols - 1), idx(re + 1, cols - 1, num_pts), idx(re, cols - 1, num_pts)))
    w_b1 = np.column_stack((idx(0, ce), idx(0, ce + 1), idx(0, ce, num_pts)))
    w_b2 = np.column_stack((idx(0, ce + 1), idx(0, ce + 1, num_pts), idx(0, ce, num_pts)))
    w_t1 = np.column_stack((idx(rows - 1, ce), idx(rows - 1, ce, num_pts), idx(rows - 1, ce + 1)))
    w_t2 = np.column_stack((idx(rows - 1, ce + 1), idx(rows - 1, ce, num_pts), idx(rows - 1, ce + 1, num_pts)))

    faces = np.vstack((t1, t2, b1, b2, w_l1, w_l2, w_r1, w_r2, w_b1, w_b2, w_t1, w_t2))

    total_w = core_w + 2 * b_w
    total_h = core_h + 2 * b_w

    boxes = [
        (0, 0, 0, b_w, total_h, b_d),
        (total_w - b_w, 0, 0, total_w, total_h, b_d),
        (b_w, 0, 0, total_w - b_w, b_w, b_d),
        (b_w, total_h - b_w, 0, total_w - b_w, total_h, b_d),
    ]

    all_v = [all_vertices]
    all_f = [faces]
    v_offset = len(all_vertices)

    for x0, y0, z0, x1, y1, z1 in boxes:
        bv, bf = _create_box(x0, y0, z0, x1, y1, z1)
        all_v.append(bv)
        all_f.append(bf + v_offset)
        v_offset += len(bv)

    # Hanging hooks for rectangle
    if manifest.hook_count > 0:
        outer_r = (manifest.hook_hole_dia_mm / 2.0) + 2.5
        hook_y = total_h + outer_r - 1.0  # Fused to top frame edge

        hook_x_positions = []
        if manifest.hook_count == 1:
            hook_x_positions.append(total_w / 2.0)
        else:
            hook_x_positions.append(b_w + (core_w * 0.2))
            hook_x_positions.append(b_w + (core_w * 0.8))

        for hx in hook_x_positions:
            hv, hf = _create_annular_hook_tab(hx, hook_y, manifest.hook_hole_dia_mm, b_d)
            all_v.append(hv)
            all_f.append(hf + v_offset)
            v_offset += len(hv)

    _save_stl(all_v, all_f, output_stl_path)
    return output_stl_path


def build_circular_litho(manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15):
    """Builds a solid circular lithophane with a raised outer ring border and integrated hook tab."""
    radius = manifest.width_mm / 2.0
    b_w = manifest.border_width_mm
    b_d = manifest.border_depth_mm
    total_radius = radius + b_w

    cols = max(10, int(round(manifest.width_mm / resolution_mm)))
    rows = max(10, int(round(manifest.height_mm / resolution_mm)))

    hmap, _, _ = generate_heightmap(
        manifest.source_image_path,
        target_width_mm=manifest.width_mm,
        target_height_mm=manifest.height_mm,
        min_thickness_mm=manifest.min_thickness_mm,
        max_thickness_mm=manifest.max_thickness_mm,
        pixel_size_mm=resolution_mm
    )

    num_rings = int(radius / resolution_mm)
    num_theta = max(48, int(2 * np.pi * total_radius / resolution_mm))

    r_vals = np.linspace(0, radius, num_rings)
    theta_vals = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    norm_x = ((x_grid + radius) / (2 * radius) * (cols - 1)).clip(0, cols - 1)
    norm_y = ((y_grid + radius) / (2 * radius) * (rows - 1)).clip(0, rows - 1)
    z_grid = hmap[norm_y.astype(int), norm_x.astype(int)]

    top_v = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    bot_v = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros_like(z_grid).ravel()))
    num_pts = top_v.shape[0]

    all_v = np.vstack((top_v, bot_v))
    faces = []

    for t in range(num_theta):
        t_next = (t + 1) % num_theta
        for r in range(num_rings - 1):
            p1 = t * num_rings + r
            p2 = t * num_rings + (r + 1)
            p3 = t_next * num_rings + (r + 1)
            p4 = t_next * num_rings + r

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

            bp1 = p1 + num_pts
            bp2 = p2 + num_pts
            bp3 = p3 + num_pts
            bp4 = p4 + num_pts
            faces.append([bp1, bp3, bp2])
            faces.append([bp1, bp4, bp3])

    # Outer ring
    border_r_vals = np.linspace(radius, total_radius, 4)
    br_grid, btheta_grid = np.meshgrid(border_r_vals, theta_vals)
    bx_grid = br_grid * np.cos(btheta_grid)
    by_grid = br_grid * np.sin(btheta_grid)
    bz_top = np.full_like(bx_grid, b_d)
    bz_bot = np.zeros_like(bx_grid)

    ring_top_v = np.column_stack((bx_grid.ravel(), by_grid.ravel(), bz_top.ravel()))
    ring_bot_v = np.column_stack((bx_grid.ravel(), by_grid.ravel(), bz_bot.ravel()))
    ring_num_pts = ring_top_v.shape[0]

    border_offset = len(all_v)
    all_v = np.vstack((all_v, ring_top_v, ring_bot_v))

    for t in range(num_theta):
        t_next = (t + 1) % num_theta
        for r in range(3):
            p1 = border_offset + (t * 4 + r)
            p2 = border_offset + (t * 4 + (r + 1))
            p3 = border_offset + (t_next * 4 + (r + 1))
            p4 = border_offset + (t_next * 4 + r)

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

            bp1 = p1 + ring_num_pts
            bp2 = p2 + ring_num_pts
            bp3 = p3 + ring_num_pts
            bp4 = p4 + ring_num_pts
            faces.append([bp1, bp3, bp2])
            faces.append([bp1, bp4, bp3])

        rim_top_1 = border_offset + (t * 4 + 3)
        rim_top_2 = border_offset + (t_next * 4 + 3)
        rim_bot_1 = rim_top_1 + ring_num_pts
        rim_bot_2 = rim_top_2 + ring_num_pts
        faces.append([rim_top_1, rim_bot_1, rim_top_2])
        faces.append([rim_top_2, rim_bot_1, rim_bot_2])

    v_list = [all_v]
    f_list = [np.array(faces, dtype=np.int32)]
    v_total = len(all_v)

    # Integrated hanging hook for circular frame
    if manifest.hook_count > 0:
        outer_r = (manifest.hook_hole_dia_mm / 2.0) + 2.5
        hook_center_y = total_radius + outer_r - 1.0
        hook_center_x = 0.0

        hv, hf = _create_annular_hook_tab(hook_center_x, hook_center_y, manifest.hook_hole_dia_mm, b_d)
        v_list.append(hv)
        f_list.append(hf + v_total)

    _save_stl(v_list, f_list, output_stl_path)
    return output_stl_path


def _create_box(x0, y0, z0, x1, y1, z1):
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
    ], dtype=np.int32)
    return v, f


def _save_stl(v_list, f_list, output_stl_path):
    all_vertices = np.vstack(v_list)
    all_faces = np.vstack(f_list)
    out_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            out_mesh.vectors[i][j] = all_vertices[face[j], :]
    out_mesh.save(output_stl_path)