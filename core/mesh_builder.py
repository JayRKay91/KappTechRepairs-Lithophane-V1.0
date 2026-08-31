import numpy as np
from stl import mesh
from core.manifest_parser import LithoManifest
from core.heightmap import generate_heightmap


def build_lithophane_stl(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """Router that handles rectangle or circular lithophanes from the web manifest."""
    if manifest.shape == "circle":
        return build_circular_litho(manifest, output_stl_path, resolution_mm)
    else:
        return build_rectangular_litho(manifest, output_stl_path, resolution_mm)


def _generate_etsy_hook_contour(
    center_x: float,
    center_y: float,
    hole_dia: float,
    anchor_y: float,
    flare_width: float = 4.5,
    num_steps: int = 40,
):
    """
    Constructs an organic Etsy-style keychain tab contour:
    1. Upper semicircular dome.
    2. Tangent concave sweeping fillets down into the frame body.
    """
    inner_r = hole_dia / 2.0
    wall_thick = 2.0
    outer_r = inner_r + wall_thick

    outer_pts = []

    # 1. Upper dome: sweep from angle 0 (right) across top (pi/2) to pi (left)
    angles_dome = np.linspace(0, np.pi, num_steps // 2)
    for a in angles_dome:
        outer_pts.append([
            center_x + outer_r * np.cos(a),
            center_y + outer_r * np.sin(a),
        ])

    # 2. Left concave flare skirt: smooth easing down to anchor_y
    n_flare = max(6, num_steps // 4)
    t_vals = np.linspace(0, 1, n_flare)
    
    # Left flare (from -outer_r to -(outer_r + flare_width))
    for t in t_vals[1:]:
        # Ease-in-out curve for natural tangent fillet
        ease = 0.5 * (1 - np.cos(np.pi * t))
        x = center_x - outer_r - (flare_width * ease)
        y = center_y - (t * (center_y - anchor_y))
        outer_pts.append([x, y])

    # 3. Base bottom anchor endpoints inside the rim body
    outer_pts.append([center_x - outer_r - flare_width, anchor_y - 0.5])
    outer_pts.append([center_x + outer_r + flare_width, anchor_y - 0.5])

    # 4. Right concave flare skirt: smooth easing back up to outer_r
    for t in reversed(t_vals[1:]):
        ease = 0.5 * (1 - np.cos(np.pi * t))
        x = center_x + outer_r + (flare_width * ease)
        y = center_y - (t * (center_y - anchor_y))
        outer_pts.append([x, y])

    outer_contour = np.array(outer_pts, dtype=np.float32)
    num_out = len(outer_contour)

    # 5. Inner circular hole
    hole_angles = np.linspace(0, 2 * np.pi, num_out, endpoint=False)
    inner_contour = np.column_stack((
        center_x + inner_r * np.cos(hole_angles),
        center_y + inner_r * np.sin(hole_angles),
    ))

    return outer_contour, inner_contour


def _create_etsy_hook_mesh(
    center_x: float,
    center_y: float,
    hole_dia: float,
    tab_depth: float,
    anchor_y: float,
    flare_width: float = 4.5,
    num_steps: int = 40,
):
    """Generates a fully closed, manifold 3D mesh for the Etsy-style hook tab."""
    outer_contour, inner_contour = _generate_etsy_hook_contour(
        center_x, center_y, hole_dia, anchor_y, flare_width, num_steps
    )
    num_out = len(outer_contour)

    top_outer = np.column_stack((outer_contour, np.full(num_out, tab_depth)))
    bot_outer = np.column_stack((outer_contour, np.zeros(num_out)))
    top_inner = np.column_stack((inner_contour, np.full(num_out, tab_depth)))
    bot_inner = np.column_stack((inner_contour, np.zeros(num_out)))

    all_v = np.vstack((top_outer, top_inner, bot_outer, bot_inner))
    faces = []

    to_off = 0
    ti_off = num_out
    bo_off = 2 * num_out
    bi_off = 3 * num_out

    for i in range(num_out):
        i_next = (i + 1) % num_out

        # Top Annular Face (+Z outward normal)
        faces.append([to_off + i, ti_off + i, to_off + i_next])
        faces.append([to_off + i_next, ti_off + i, ti_off + i_next])

        # Bottom Annular Face (-Z outward normal)
        faces.append([bo_off + i, bo_off + i_next, bi_off + i])
        faces.append([bo_off + i_next, bi_off + i_next, bi_off + i])

        # Inner hole vertical cylinder wall
        faces.append([ti_off + i, bi_off + i, ti_off + i_next])
        faces.append([ti_off + i_next, bi_off + i, bi_off + i_next])

        # Outer filleted vertical wall
        faces.append([to_off + i, to_off + i_next, bo_off + i])
        faces.append([to_off + i_next, bo_off + i_next, bo_off + i])

    # Seal base anchor edge
    idx_base_l = (num_steps // 2) + (num_steps // 4)
    idx_base_r = idx_base_l + 1

    p_tl = to_off + idx_base_l
    p_tr = to_off + idx_base_r
    p_bl = bo_off + idx_base_l
    p_br = bo_off + idx_base_r

    faces.append([p_tl, p_tr, p_bl])
    faces.append([p_tr, p_br, p_bl])

    return all_v, np.array(faces, dtype=np.int32)


def build_circular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """
    Constructs a solid monolithic circular lithophane with an integrated
    Etsy-style swept filleted hanging tab.
    """
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
        pixel_size_mm=resolution_mm,
    )
    hmap = np.flipud(hmap)

    num_photo_rings = max(12, int(round(radius / resolution_mm)))
    num_border_rings = max(4, int(round(b_w / resolution_mm)))
    total_rings = num_photo_rings + num_border_rings

    num_theta = max(72, int(round(2 * np.pi * total_radius / resolution_mm)))

    r_photo = np.linspace(0, radius, num_photo_rings, endpoint=False)
    r_border = np.linspace(radius, total_radius, num_border_rings)
    r_vals = np.concatenate((r_photo, r_border))

    theta_vals = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    z_grid = np.zeros_like(r_grid)
    norm_x = ((x_grid + radius) / (2 * radius) * (cols - 1)).clip(0, cols - 1)
    norm_y = ((y_grid + radius) / (2 * radius) * (rows - 1)).clip(0, rows - 1)
    photo_z = hmap[norm_y.astype(int), norm_x.astype(int)]

    mask_photo = r_grid < radius
    z_grid[mask_photo] = photo_z[mask_photo]
    z_grid[~mask_photo] = b_d

    top_v = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    bot_v = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros_like(z_grid).ravel()))
    num_pts = top_v.shape[0]

    all_v = [np.vstack((top_v, bot_v))]
    faces = []

    for t in range(num_theta):
        t_next = (t + 1) % num_theta
        for r in range(total_rings - 1):
            p1 = t * total_rings + r
            p2 = t * total_rings + (r + 1)
            p3 = t_next * total_rings + (r + 1)
            p4 = t_next * total_rings + r

            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

            bp1 = p1 + num_pts
            bp2 = p2 + num_pts
            bp3 = p3 + num_pts
            bp4 = p4 + num_pts
            faces.append([bp1, bp3, bp2])
            faces.append([bp1, bp4, bp3])

        rim_top_1 = t * total_rings + (total_rings - 1)
        rim_top_2 = t_next * total_rings + (total_rings - 1)
        rim_bot_1 = rim_top_1 + num_pts
        rim_bot_2 = rim_top_2 + num_pts
        faces.append([rim_top_1, rim_bot_1, rim_top_2])
        faces.append([rim_top_2, rim_bot_1, rim_bot_2])

    all_f = [np.array(faces, dtype=np.int32)]
    v_total = len(all_v[0])

    if manifest.hook_count > 0:
        hole_dia = manifest.hook_hole_dia_mm
        inner_r = hole_dia / 2.0
        wall_thick = 2.0
        outer_r = inner_r + wall_thick
        
        # Position dome so hole sits cleanly above the rim with flared skirt anchored into the border
        hook_cy = total_radius + inner_r + 1.0
        hook_cx = 0.0
        anchor_y = total_radius - (b_w * 0.8)

        hv, hf = _create_etsy_hook_mesh(
            center_x=hook_cx,
            center_y=hook_cy,
            hole_dia=hole_dia,
            tab_depth=b_d,
            anchor_y=anchor_y,
            flare_width=4.5,
        )
        all_v.append(hv)
        all_f.append(hf + v_total)

    _save_stl(all_v, all_f, output_stl_path)
    return output_stl_path


def build_rectangular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """Builds a solid rectangular lithophane with borders and Etsy-style swept hooks."""
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
        pixel_size_mm=resolution_mm,
    )
    hmap = np.flipud(hmap)

    x = np.linspace(b_w, b_w + core_w, cols)
    y = np.linspace(b_w, b_w + core_h, rows)
    xx, yy = np.meshgrid(x, y)

    top_v = np.column_stack((xx.ravel(), yy.ravel(), hmap.ravel()))
    bot_v = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(hmap).ravel()))
    num_pts = rows * cols
    all_vertices = np.vstack((top_v, bot_v))

    def idx(r, c, offset=0):
        return offset + (r * cols + c)

    r_idx, c_idx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    rf = r_idx.ravel()
    cf = c_idx.ravel()

    t1 = np.column_stack((idx(rf, cf), idx(rf + 1, cf), idx(rf + 1, cf + 1)))
    t2 = np.column_stack((idx(rf, cf), idx(rf + 1, cf + 1), idx(rf, cf + 1)))
    b1 = np.column_stack((idx(rf, cf, num_pts), idx(rf + 1, cf + 1, num_pts), idx(rf + 1, cf, num_pts)))
    b2 = np.column_stack((idx(rf, cf, num_pts), idx(rf, cf + 1, num_pts), idx(rf + 1, cf + 1, num_pts)))

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

    if manifest.hook_count > 0:
        hole_dia = manifest.hook_hole_dia_mm
        inner_r = hole_dia / 2.0
        wall_thick = 2.0
        outer_r = inner_r + wall_thick

        hook_y = total_h + inner_r + 1.0
        anchor_y = total_h - (b_w * 0.8)

        hook_x_positions = []
        if manifest.hook_count == 1:
            hook_x_positions.append(total_w / 2.0)
        else:
            hook_x_positions.append(b_w + (core_w * 0.2))
            hook_x_positions.append(b_w + (core_w * 0.8))

        for hx in hook_x_positions:
            hv, hf = _create_etsy_hook_mesh(
                center_x=hx,
                center_y=hook_y,
                hole_dia=hole_dia,
                tab_depth=b_d,
                anchor_y=anchor_y,
                flare_width=4.5,
            )
            all_v.append(hv)
            all_f.append(hf + v_offset)
            v_offset += len(hv)

    _save_stl(all_v, all_f, output_stl_path)
    return output_stl_path


def _create_box(x0, y0, z0, x1, y1, z1):
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5]
    ], dtype=np.int32)
    return v, f


def _save_stl(v_list, f_list, output_stl_path):
    all_vertices = np.vstack(v_list)
    all_faces = np.vstack(f_list)

    out_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            out_mesh.vectors[i][j] = all_vertices[face[j], :]

    out_mesh.update_normals()
    out_mesh.save(output_stl_path)