import numpy as np
import trimesh
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


def _create_hook_tab_mesh(
    center_x: float,
    center_y: float,
    hole_dia: float,
    tab_depth: float,
    anchor_y: float,
    num_steps: int = 36,
) -> trimesh.Trimesh:
    """
    Creates an independent, fully manifold watertight Trimesh object
    for the filleted eyelet tab.
    """
    inner_r = hole_dia / 2.0
    wall_thick = 2.5
    outer_r = inner_r + wall_thick
    fillet_flare = 3.5

    outer_pts = []

    # Upper dome sweep: angle 0 to pi
    angles_dome = np.linspace(0, np.pi, num_steps // 2)
    for a in angles_dome:
        outer_pts.append([
            center_x + outer_r * np.cos(a),
            center_y + outer_r * np.sin(a),
        ])

    # Flare down to anchor inside the frame rim
    outer_pts.append([center_x - outer_r - (fillet_flare * 0.4), center_y - (outer_r * 0.4)])
    outer_pts.append([center_x - outer_r - fillet_flare, anchor_y])
    outer_pts.append([center_x + outer_r + fillet_flare, anchor_y])
    outer_pts.append([center_x + outer_r + (fillet_flare * 0.4), center_y - (outer_r * 0.4)])

    outer_contour = np.array(outer_pts, dtype=np.float32)
    num_out = len(outer_contour)

    # Inner circular opening (CCW)
    hole_angles = np.linspace(0, 2 * np.pi, num_out, endpoint=False)
    inner_contour = np.column_stack((
        center_x + inner_r * np.cos(hole_angles),
        center_y + inner_r * np.sin(hole_angles),
    ))

    # 3D vertices
    top_outer = np.column_stack((outer_contour, np.full(num_out, tab_depth)))
    bot_outer = np.column_stack((outer_contour, np.zeros(num_out)))
    top_inner = np.column_stack((inner_contour, np.full(num_out, tab_depth)))
    bot_inner = np.column_stack((inner_contour, np.zeros(num_out)))

    vertices = np.vstack((top_outer, top_inner, bot_outer, bot_inner))
    faces = []

    to_off = 0
    ti_off = num_out
    bo_off = 2 * num_out
    bi_off = 3 * num_out

    for i in range(num_out):
        i_next = (i + 1) % num_out

        # Top Annular Face (+Z)
        faces.append([to_off + i, ti_off + i, to_off + i_next])
        faces.append([to_off + i_next, ti_off + i, ti_off + i_next])

        # Bottom Annular Face (-Z)
        faces.append([bo_off + i, bo_off + i_next, bi_off + i])
        faces.append([bo_off + i_next, bi_off + i_next, bi_off + i])

        # Inner hole cylinder wall
        faces.append([ti_off + i, bi_off + i, ti_off + i_next])
        faces.append([ti_off + i_next, bi_off + i, bi_off + i_next])

        # Outer filleted wall
        faces.append([to_off + i, to_off + i_next, bo_off + i])
        faces.append([to_off + i_next, bo_off + i_next, bo_off + i])

    # Endcap wall on the base anchor edge
    idx_base_left = (num_steps // 2) + 1
    idx_base_right = (num_steps // 2) + 2

    p_tl = to_off + idx_base_left
    p_tr = to_off + idx_base_right
    p_bl = bo_off + idx_base_left
    p_br = bo_off + idx_base_right

    faces.append([p_tl, p_tr, p_bl])
    faces.append([p_tr, p_br, p_bl])

    tab_mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32), process=True)
    return tab_mesh


def build_circular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """
    Constructs a solid circular lithophane, builds the hook tab,
    and fuses them with a true CSG boolean union to produce a single-shell STL.
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

    # Invert rows so row 0 (+Y) matches the top of the 3D print
    hmap = np.flipud(hmap)

    num_photo_rings = max(12, int(round(radius / resolution_mm)))
    num_border_rings = max(4, int(round(b_w / resolution_mm)))
    total_rings = num_photo_rings + num_border_rings

    num_theta = max(64, int(round(2 * np.pi * total_radius / resolution_mm)))

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

    all_v = np.vstack((top_v, bot_v))
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

    base_mesh = trimesh.Trimesh(vertices=all_v, faces=np.array(faces, dtype=np.int32), process=True)

    if manifest.hook_count > 0:
        inner_r = manifest.hook_hole_dia_mm / 2.0
        outer_r = inner_r + 2.5
        hook_center_y = total_radius + outer_r - 1.0
        hook_center_x = 0.0
        anchor_y = total_radius - (b_w * 0.8)

        tab_mesh = _create_hook_tab_mesh(
            center_x=hook_center_x,
            center_y=hook_center_y,
            hole_dia=manifest.hook_hole_dia_mm,
            tab_depth=b_d,
            anchor_y=anchor_y,
        )

        try:
            # CSG boolean union using manifold engine
            final_mesh = base_mesh.union(tab_mesh, engine="manifold")
        except Exception:
            # Fallback to standard union if manifold encounters planar overlaps
            final_mesh = trimesh.boolean.union([base_mesh, tab_mesh])
    else:
        final_mesh = base_mesh

    final_mesh.export(output_stl_path)
    return output_stl_path


def build_rectangular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """Builds a solid rectangular lithophane with borders and fused hooks."""
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

    photo_faces = np.vstack((t1, t2, b1, b2, w_l1, w_l2, w_r1, w_r2, w_b1, w_b2, w_t1, w_t2))
    photo_mesh = trimesh.Trimesh(vertices=all_vertices, faces=photo_faces, process=True)

    total_w = core_w + 2 * b_w
    total_h = core_h + 2 * b_w

    # Outer border rails
    border_boxes = [
        trimesh.creation.box(bounds=[[0, 0, 0], [b_w, total_h, b_d]]),
        trimesh.creation.box(bounds=[[total_w - b_w, 0, 0], [total_w, total_h, b_d]]),
        trimesh.creation.box(bounds=[[b_w, 0, 0], [total_w - b_w, b_w, b_d]]),
        trimesh.creation.box(bounds=[[b_w, total_h - b_w, 0], [total_w - b_w, total_h, b_d]]),
    ]

    parts = [photo_mesh] + border_boxes

    if manifest.hook_count > 0:
        inner_r = manifest.hook_hole_dia_mm / 2.0
        outer_r = inner_r + 2.5
        hook_y = total_h + outer_r - 1.5

        hook_x_positions = []
        if manifest.hook_count == 1:
            hook_x_positions.append(total_w / 2.0)
        else:
            hook_x_positions.append(b_w + (core_w * 0.2))
            hook_x_positions.append(b_w + (core_w * 0.8))

        for hx in hook_x_positions:
            parts.append(
                _create_hook_tab_mesh(
                    center_x=hx,
                    center_y=hook_y,
                    hole_dia=manifest.hook_hole_dia_mm,
                    tab_depth=b_d,
                    anchor_y=total_h - 1.5,
                )
            )

    try:
        final_mesh = trimesh.boolean.union(parts, engine="manifold")
    except Exception:
        final_mesh = trimesh.boolean.union(parts)

    final_mesh.export(output_stl_path)
    return output_stl_path