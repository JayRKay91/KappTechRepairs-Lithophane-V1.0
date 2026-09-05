# === Litho Mesh Studio - Version 1.0 Final Release ===

import numpy as np
from stl import mesh
from core.manifest_parser import LithoManifest
from core.heightmap import generate_heightmap


def build_lithophane_stl(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    if manifest.shape == "circle":
        return build_circular_litho(manifest, output_stl_path, resolution_mm)
    else:
        return build_rectangular_litho(manifest, output_stl_path, resolution_mm)


def create_hook_tab(
    center_x: float,
    center_y: float,
    hole_dia_mm: float,
    depth_mm: float,
    rim_edge_y: float,
    wall_thickness: float = 2.5,
    n_steps: int = 36,
):
    hole_rad = hole_dia_mm / 2.0
    outer_rad = hole_rad + wall_thickness

    angles = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    t_r_vals = np.linspace(hole_rad, outer_rad, 4)
    tr_grid, tan_grid = np.meshgrid(t_r_vals, angles)

    tx = center_x + tr_grid * np.cos(tan_grid)
    ty = center_y + tr_grid * np.sin(tan_grid)

    mask_base = (tan_grid > np.pi * 1.1) & (tan_grid < np.pi * 1.9)
    ty[mask_base] = np.minimum(ty[mask_base], rim_edge_y)

    tab_top_v = np.column_stack((tx.ravel(), ty.ravel(), np.full(tx.size, depth_mm)))
    tab_bot_v = np.column_stack((tx.ravel(), ty.ravel(), np.zeros(tx.size)))
    tab_pts = tab_top_v.shape[0]

    tab_all_v = np.vstack((tab_top_v, tab_bot_v))
    tab_faces = []

    for i in range(n_steps):
        i_next = (i + 1) % n_steps
        for j in range(3):
            p1 = i * 4 + j
            p2 = i * 4 + (j + 1)
            p3 = i_next * 4 + (j + 1)
            p4 = i_next * 4 + j

            tab_faces.append([p1, p2, p3])
            tab_faces.append([p1, p3, p4])

            bp1 = p1 + tab_pts
            bp2 = p2 + tab_pts
            bp3 = p3 + tab_pts
            bp4 = p4 + tab_pts
            tab_faces.append([bp1, bp3, bp2])
            tab_faces.append([bp1, bp4, bp3])

        in_t1 = i * 4
        in_t2 = i_next * 4
        in_b1 = in_t1 + tab_pts
        in_b2 = in_t2 + tab_pts
        tab_faces.append([in_t1, in_t2, in_b1])
        tab_faces.append([in_t2, in_b2, in_b1])

        out_t1 = i * 4 + 3
        out_t2 = i_next * 4 + 3
        out_b1 = out_t1 + tab_pts
        out_b2 = out_t2 + tab_pts
        tab_faces.append([out_t1, out_b1, out_t2])
        tab_faces.append([out_t2, out_b1, out_b2])

    return tab_all_v, np.array(tab_faces, dtype=np.int32)


def build_circular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
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
        outer_rad = (manifest.hook_hole_dia_mm / 2.0) + 2.5
        hook_cy = total_radius + outer_rad - 1.0
        hook_v, hook_f = create_hook_tab(
            center_x=0.0,
            center_y=hook_cy,
            hole_dia_mm=manifest.hook_hole_dia_mm,
            depth_mm=b_d,
            rim_edge_y=total_radius - 0.5,
        )
        all_v.append(hook_v)
        all_f.append(hook_f + v_total)

    _save_stl(all_v, all_f, output_stl_path)
    return output_stl_path


def build_rectangular_litho(
    manifest: LithoManifest, output_stl_path: str, resolution_mm: float = 0.15
):
    """
    Constructs a solid rectangular lithophane using identical quad stitching
    and outward winding as the circle generator.
    """
    core_w = manifest.width_mm
    core_h = manifest.height_mm
    b_w = manifest.border_width_mm
    b_d = manifest.border_depth_mm

    hmap, img_rows, img_cols = generate_heightmap(
        manifest.source_image_path,
        target_width_mm=core_w,
        target_height_mm=core_h,
        min_thickness_mm=manifest.min_thickness_mm,
        max_thickness_mm=manifest.max_thickness_mm,
        pixel_size_mm=resolution_mm,
    )
    hmap = np.flipud(hmap)

    b_px = max(2, int(round(b_w / resolution_mm)))
    total_rows = img_rows + 2 * b_px
    total_cols = img_cols + 2 * b_px

    full_z = np.full((total_rows, total_cols), b_d, dtype=np.float32)
    full_z[b_px : b_px + img_rows, b_px : b_px + img_cols] = hmap

    total_w = core_w + 2 * b_w
    total_h = core_h + 2 * b_w

    x = np.linspace(0, total_w, total_cols)
    y = np.linspace(0, total_h, total_rows)
    xx, yy = np.meshgrid(x, y)

    top_v = np.column_stack((xx.ravel(), yy.ravel(), full_z.ravel()))
    bot_v = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(full_z).ravel()))
    num_pts = total_rows * total_cols
    all_vertices = np.vstack((top_v, bot_v))

    faces = []

    # Identical quad stitching loop to the circle generator
    for r in range(total_rows - 1):
        for c in range(total_cols - 1):
            p1 = r * total_cols + c
            p2 = r * total_cols + (c + 1)
            p3 = (r + 1) * total_cols + (c + 1)
            p4 = (r + 1) * total_cols + c

            # Top faces (Z+)
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

            # Bottom faces (Z-)
            bp1 = p1 + num_pts
            bp2 = p2 + num_pts
            bp3 = p3 + num_pts
            bp4 = p4 + num_pts
            faces.append([bp1, bp3, bp2])
            faces.append([bp1, bp4, bp3])

    # 4 Outer Perimeter Walls
    # Bottom wall (r = 0)
    for c in range(total_cols - 1):
        t1, t2 = c, c + 1
        b1, b2 = t1 + num_pts, t2 + num_pts
        faces.append([t1, b1, t2])
        faces.append([t2, b1, b2])

    # Top wall (r = total_rows - 1)
    for c in range(total_cols - 1):
        t1 = (total_rows - 1) * total_cols + c
        t2 = t1 + 1
        b1, b2 = t1 + num_pts, t2 + num_pts
        faces.append([t1, t2, b1])
        faces.append([t2, b2, b1])

    # Left wall (c = 0)
    for r in range(total_rows - 1):
        t1 = r * total_cols
        t2 = (r + 1) * total_cols
        b1, b2 = t1 + num_pts, t2 + num_pts
        faces.append([t1, t2, b1])
        faces.append([t2, b2, b1])

    # Right wall (c = total_cols - 1)
    for r in range(total_rows - 1):
        t1 = r * total_cols + (total_cols - 1)
        t2 = (r + 1) * total_cols + (total_cols - 1)
        b1, b2 = t1 + num_pts, t2 + num_pts
        faces.append([t1, b1, t2])
        faces.append([t2, b1, b2])

    all_v = [all_vertices]
    all_f = [np.array(faces, dtype=np.int32)]
    v_total = len(all_vertices)

    # Add hooks
    if manifest.hook_count > 0:
        outer_rad = (manifest.hook_hole_dia_mm / 2.0) + 2.5
        hook_cy = total_h + outer_rad - 1.0

        if manifest.hook_count == 1:
            hook_x_positions = [total_w / 2.0]
        else:
            hook_x_positions = [
                total_w * (i + 1) / (manifest.hook_count + 1)
                for i in range(manifest.hook_count)
            ]

        for hx in hook_x_positions:
            hv, hf = create_hook_tab(
                center_x=hx,
                center_y=hook_cy,
                hole_dia_mm=manifest.hook_hole_dia_mm,
                depth_mm=b_d,
                rim_edge_y=total_h - 0.5,
            )
            all_v.append(hv)
            all_f.append(hf + v_total)
            v_total += len(hv)

    _save_stl(all_v, all_f, output_stl_path)
    return output_stl_path


def _save_stl(v_list, f_list, output_stl_path):
    all_vertices = np.vstack(v_list)
    all_faces = np.vstack(f_list)

    out_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            out_mesh.vectors[i][j] = all_vertices[face[j], :]

    out_mesh.update_normals()
    out_mesh.save(output_stl_path)