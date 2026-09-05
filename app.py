# === Litho Mesh Studio - Version 1.0 Final Release ===

import os
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from core.manifest_parser import parse_manifest, LithoManifest
from core.mesh_builder import build_lithophane_stl

# Configure CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LithoMeshStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Kapp Tech Repairs - Litho Mesh Studio V1.0")
        self.geometry("780x620")
        self.minsize(700, 560)

        self.manifest: LithoManifest = None
        self.manifest_path = None
        self.last_generated_stl = None

        self._build_ui()

    def _build_ui(self):
        # Header banner
        self.header_frame = ctk.CTkFrame(self, corner_radius=8)
        self.header_frame.pack(fill="x", padx=16, pady=(16, 8))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Litho Mesh Studio V1.0",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(side="left", padx=16, pady=12)

        # File selection bar
        self.file_frame = ctk.CTkFrame(self, corner_radius=8)
        self.file_frame.pack(fill="x", padx=16, pady=8)

        self.file_entry = ctk.CTkEntry(
            self.file_frame,
            placeholder_text="Select a manifest JSON file...",
            font=ctk.CTkFont(size=12)
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(16, 8), pady=12)

        self.browse_btn = ctk.CTkButton(
            self.file_frame,
            text="Browse JSON",
            width=110,
            command=self._browse_manifest
        )
        self.browse_btn.pack(side="right", padx=(0, 16), pady=12)

        # Main details panel (Left: Order Info, Right: Geometry Specs)
        self.content_frame = ctk.CTkFrame(self, corner_radius=8)
        self.content_frame.pack(fill="both", expand=True, padx=16, pady=8)
        self.content_frame.grid_columnconfigure((0, 1), weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # Left Column: Metadata & Notes
        self.meta_box = ctk.CTkTextbox(self.content_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.meta_box.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.meta_box.insert("end", "Waiting for manifest file...\n\nLoad a .json manifest exported from the customizer webpage to inspect order parameters.")
        self.meta_box.configure(state="disabled")

        # Right Column: 3D Geometry Readout
        self.specs_box = ctk.CTkTextbox(self.content_frame, font=ctk.CTkFont(family="Consolas", size=12))
        self.specs_box.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.specs_box.insert("end", "Geometry parameters will appear here.")
        self.specs_box.configure(state="disabled")

        # Actions & progress panel
        self.action_frame = ctk.CTkFrame(self, corner_radius=8)
        self.action_frame.pack(fill="x", padx=16, pady=(8, 16))

        # Resolution dropdown
        self.res_label = ctk.CTkLabel(self.action_frame, text="Resolution:", font=ctk.CTkFont(size=12))
        self.res_label.pack(side="left", padx=(16, 6), pady=12)

        self.res_var = ctk.StringVar(value="Production (0.15mm)")
        self.res_menu = ctk.CTkOptionMenu(
            self.action_frame,
            values=["Ultra Detail (0.10mm)", "Production (0.15mm)", "Fast Draft (0.25mm)"],
            variable=self.res_var,
            width=160
        )
        self.res_menu.pack(side="left", padx=(0, 12), pady=12)

        # Generate STL Button
        self.generate_btn = ctk.CTkButton(
            self.action_frame,
            text="Generate STL",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            command=self._start_generation,
            state="disabled"
        )
        self.generate_btn.pack(side="right", padx=(8, 16), pady=12)

        # Open in Slicer Button
        self.slicer_btn = ctk.CTkButton(
            self.action_frame,
            text="Open in Lychee",
            fg_color="#17a2b8",
            hover_color="#138496",
            command=self._open_in_slicer,
            state="disabled"
        )
        self.slicer_btn.pack(side="right", padx=(8, 0), pady=12)

        # Status indicator
        self.status_label = ctk.CTkLabel(self, text="Ready", font=ctk.CTkFont(size=11), text_color="gray")
        self.status_label.pack(side="bottom", anchor="w", padx=20, pady=(0, 6))

    def _browse_manifest(self):
        file_path = filedialog.askopenfilename(
            title="Select Lithophane Manifest",
            filetypes=[("JSON Files", "*.json")]
        )
        if not file_path:
            return

        self.manifest_path = file_path
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, file_path)

        try:
            self.manifest = parse_manifest(file_path)
            self._update_readouts()
            self.generate_btn.configure(state="normal")
            self.status_label.configure(text=f"Loaded manifest: {os.path.basename(file_path)}", text_color="#28a745")
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse manifest:\n{str(e)}")
            self.generate_btn.configure(state="disabled")
            self.status_label.configure(text="Error loading manifest", text_color="#dc3545")

    def _update_readouts(self):
        m = self.manifest

        meta_text = (
            "--- CLIENT & ORDER ---\n"
            f"Client Name:  {m.client_name}\n"
            f"Client Email: {m.client_email}\n"
            f"Order Date:   {m.order_date}\n\n"
            "--- SPECIAL NOTES ---\n"
            f"{m.special_notes if m.special_notes else '(None)'}\n\n"
            "--- SOURCE IMAGE ---\n"
            f"Filename: {os.path.basename(m.source_image_path)}\n"
            f"Exists:   {'YES' if os.path.exists(m.source_image_path) else 'MISSING!'}"
        )

        specs_text = (
            "--- 3D GEOMETRY SPECS ---\n"
            f"Shape:            {m.shape.upper()}\n"
            f"Dimensions:       {m.width_mm:.1f} x {m.height_mm:.1f} mm\n"
            f"Min Thickness:    {m.min_thickness_mm:.2f} mm\n"
            f"Max Thickness:    {m.max_thickness_mm:.2f} mm\n"
            f"Border Width:     {m.border_width_mm:.1f} mm\n"
            f"Border Depth:     {m.border_depth_mm:.1f} mm\n\n"
            "--- HANGING EYELETS ---\n"
            f"Hook Count:       {m.hook_count}\n"
            f"Hole Diameter:    {m.hook_hole_dia_mm:.1f} mm\n"
            f"Position:         {', '.join(m.hook_positions) if m.hook_positions else 'None'}"
        )

        self.meta_box.configure(state="normal")
        self.meta_box.delete("1.0", "end")
        self.meta_box.insert("end", meta_text)
        self.meta_box.configure(state="disabled")

        self.specs_box.configure(state="normal")
        self.specs_box.delete("1.0", "end")
        self.specs_box.insert("end", specs_text)
        self.specs_box.configure(state="disabled")

    def _start_generation(self):
        if not self.manifest:
            return

        # Determine chosen resolution
        res_choice = self.res_var.get()
        if "0.10mm" in res_choice:
            res_mm = 0.10
        elif "0.25mm" in res_choice:
            res_mm = 0.25
        else:
            res_mm = 0.15

        base_dir = os.path.dirname(os.path.abspath(self.manifest_path))
        base_name = os.path.splitext(os.path.basename(self.manifest_path))[0].replace("_manifest", "")
        output_stl = os.path.join(base_dir, f"{base_name}_print_ready.stl")

        self.generate_btn.configure(state="disabled")
        self.browse_btn.configure(state="disabled")
        self.status_label.configure(text="Generating watertight solid mesh... Please wait.", text_color="#17a2b8")

        def worker():
            try:
                build_lithophane_stl(self.manifest, output_stl, resolution_mm=res_mm)
                self.last_generated_stl = output_stl
                self.after(0, lambda: self._on_generation_success(output_stl))
            except Exception as e:
                self.after(0, lambda: self._on_generation_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generation_success(self, stl_path):
        self.generate_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")
        self.slicer_btn.configure(state="normal")
        self.status_label.configure(text=f"STL Created: {os.path.basename(stl_path)}", text_color="#28a745")
        messagebox.showinfo("Success", f"Watertight STL generated successfully!\n\nSaved to:\n{stl_path}")

    def _on_generation_error(self, err_msg):
        self.generate_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")
        self.status_label.configure(text="Generation failed.", text_color="#dc3545")
        messagebox.showerror("Generation Error", f"Mesh generation failed:\n{err_msg}")

    def _open_in_slicer(self):
        if not self.last_generated_stl or not os.path.exists(self.last_generated_stl):
            messagebox.showwarning("File Missing", "No generated STL found to launch.")
            return

        try:
            # On Windows, os.startfile opens with the associated program (e.g., Lychee Slicer)
            os.startfile(self.last_generated_stl)
        except Exception as e:
            messagebox.showerror("Launch Error", f"Could not launch slicer:\n{str(e)}")


if __name__ == "__main__":
    app = LithoMeshStudioApp()
    app.mainloop()