# === Litho Mesh Studio - Version 1.0 Final Release ===

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LithoManifest:
    # Metadata
    client_name: str
    client_email: str
    order_date: str
    special_notes: str

    # Shape & Dimensions
    shape: str
    width_mm: float
    height_mm: float
    border_width_mm: float
    border_depth_mm: float
    min_thickness_mm: float
    max_thickness_mm: float

    # Resolution
    pixel_width: int
    pixel_height: int
    target_dpi: int

    # Hooks
    hook_count: int
    hook_hole_dia_mm: float
    hook_positions: List[str]
    hook_tab_style: str

    # Image Tone & Path
    contrast_multiplier: float
    brightness_multiplier: float
    source_image_path: str


def parse_manifest(json_filepath: str) -> LithoManifest:
    """Parses and validates the exact web customizer JSON schema."""
    if not os.path.exists(json_filepath):
        raise FileNotFoundError(f"Manifest file not found: {json_filepath}")

    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest_dir = os.path.dirname(os.path.abspath(json_filepath))

    metadata = data.get("metadata", {})
    params = data.get("lithophaneParameters", {})
    dims = params.get("dimensionsMM", {})
    res = params.get("exportResolution", {})
    hooks = params.get("integratedHangingHooks", {})
    tone = params.get("imageTone", {})

    # Resolve image path relative to the manifest directory
    source_filename = params.get("sourceFile", "")
    if not source_filename:
        raise ValueError("Missing 'sourceFile' inside lithophaneParameters.")

    if os.path.isabs(source_filename):
        resolved_img = source_filename
    else:
        resolved_img = os.path.join(manifest_dir, source_filename)

    return LithoManifest(
        client_name=metadata.get("clientName", "Unknown"),
        client_email=metadata.get("clientEmail", ""),
        order_date=metadata.get("orderDate", ""),
        special_notes=metadata.get("specialNotes", ""),
        shape=params.get("shape", "rectangle").strip().lower(),
        width_mm=float(dims.get("width", 100.0)),
        height_mm=float(dims.get("height", 100.0)),
        border_width_mm=float(dims.get("borderWidth", 5.0)),
        border_depth_mm=float(dims.get("borderDepth", 4.0)),
        min_thickness_mm=float(dims.get("imageThinnestLayer", 0.8)),
        max_thickness_mm=float(dims.get("imageThickestLayer", 3.0)),
        pixel_width=int(res.get("pixelWidth", 900)),
        pixel_height=int(res.get("pixelHeight", 900)),
        target_dpi=int(res.get("targetDPI", 300)),
        hook_count=int(hooks.get("count", 0)),
        hook_hole_dia_mm=float(hooks.get("holeDiameterMM", 4.0)),
        hook_positions=hooks.get("positions", []),
        hook_tab_style=hooks.get("outerTabStyle", "etsy_flush_fillet"),
        contrast_multiplier=float(tone.get("contrastMultiplier", 1.0)),
        brightness_multiplier=float(tone.get("brightnessMultiplier", 1.0)),
        source_image_path=resolved_img,
    )