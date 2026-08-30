import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LithoManifest:
    shape: str
    width_mm: float
    height_mm: float
    diameter_mm: float
    min_thickness_mm: float
    max_thickness_mm: float
    border_width_mm: float
    border_thickness_mm: float
    has_hooks: bool
    hook_hole_dia_mm: float
    hook_tab_width_mm: float
    image_path: str
    order_id: Optional[str] = None


def parse_manifest(json_filepath: str) -> LithoManifest:
    """
    Parses a manifest JSON file from the web customizer.
    Validates required fields, applies safe resin-printing defaults,
    and resolves the associated image path.
    """
    if not os.path.exists(json_filepath):
        raise FileNotFoundError(f"Manifest file not found: {json_filepath}")

    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest_dir = os.path.dirname(os.path.abspath(json_filepath))

    # Resolve image path relative to the manifest directory if not absolute
    raw_img_path = data.get("image_path") or data.get("image_filename")
    if not raw_img_path:
        raise ValueError("Manifest must specify 'image_path' or 'image_filename'.")

    if os.path.isabs(raw_img_path):
        resolved_img = raw_img_path
    else:
        resolved_img = os.path.join(manifest_dir, raw_img_path)

    if not os.path.exists(resolved_img):
        raise FileNotFoundError(f"Referenced image not found: {resolved_img}")

    # Extract geometry dimensions with production defaults
    shape = str(data.get("shape", "rectangle")).strip().lower()
    width_mm = float(data.get("width_mm", 100.0))
    height_mm = float(data.get("height_mm", 100.0))
    diameter_mm = float(data.get("diameter_mm", max(width_mm, height_mm)))

    # Resin printing thickness defaults (0.8mm min for translucency, 2.8mm max for contrast)
    min_thickness = float(data.get("min_thickness_mm", 0.8))
    max_thickness = float(data.get("max_thickness_mm", 2.8))

    # Outer structural frame defaults
    border_width = float(data.get("border_width_mm", 5.0))
    border_thickness = float(data.get("border_thickness_mm", max_thickness))

    # Integrated hanging tabs
    has_hooks = bool(data.get("has_hooks", False))
    hook_hole_dia = float(data.get("hook_hole_dia_mm", 3.0))
    hook_tab_width = float(data.get("hook_tab_width_mm", 7.0))

    order_id = data.get("order_id", None)

    return LithoManifest(
        shape=shape,
        width_mm=width_mm,
        height_mm=height_mm,
        diameter_mm=diameter_mm,
        min_thickness_mm=min_thickness,
        max_thickness_mm=max_thickness,
        border_width_mm=border_width,
        border_thickness_mm=border_thickness,
        has_hooks=has_hooks,
        hook_hole_dia_mm=hook_hole_dia,
        hook_tab_width_mm=hook_tab_width,
        image_path=resolved_img,
        order_id=order_id,
    )