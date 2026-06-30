"""
gui.py -- minimal Tkinter control panel for the tablet-half alignment methods.

Pick the two PLY files (A = fixed/target, B = moving/source), run either or both
methods, read the fitness/rmse, then click "View" to open an Open3D window and
eyeball the result. Target is green, source is red; good alignment = the two
colors interleave through the overlap. "Save merged" writes a single combined PLY.

Run:  ./venv/bin/python gui.py
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import open3d as o3d
import align


class App:
    def __init__(self, root):
        self.root = root
        root.title("Tablet half alignment")
        root.geometry("760x560")

        self.q = queue.Queue()
        self.A = None          # MeshData target
        self.B = None          # MeshData source
        self.results = {}      # method name -> AlignResult
        self.busy = False

        pad = dict(padx=6, pady=4)
        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)

        # ---- file pickers ----
        self.pathA = tk.StringVar()
        self.pathB = tk.StringVar()
        ttk.Label(frm, text="Mesh A  (fixed / target)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.pathA, width=64).grid(row=0, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._browse(self.pathA)).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Mesh B  (moving / source)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.pathB, width=64).grid(row=1, column=1, **pad)
        ttk.Button(frm, text="Browse", command=lambda: self._browse(self.pathB)).grid(row=1, column=2, **pad)

        # ---- params ----
        self.voxel = tk.StringVar(value="auto")
        self.nsamp = tk.StringVar(value="60000")
        prm = ttk.Frame(frm)
        prm.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(prm, text="voxel:").pack(side="left")
        ttk.Entry(prm, textvariable=self.voxel, width=10).pack(side="left", padx=(2, 14))
        ttk.Label(prm, text="sample pts:").pack(side="left")
        ttk.Entry(prm, textvariable=self.nsamp, width=10).pack(side="left", padx=2)

        # ---- run buttons ----
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(btns, text="Load meshes", command=self.load).pack(side="left", padx=4)
        ttk.Button(btns, text="Opening", command=lambda: self.run("opening")).pack(side="left", padx=4)
        ttk.Button(btns, text="FPFH", command=lambda: self.run("fpfh")).pack(side="left", padx=4)
        ttk.Button(btns, text="Collapse+Flip", command=lambda: self.run("collapse")).pack(side="left", padx=4)
        ttk.Button(btns, text="Run All", command=lambda: self.run("all")).pack(side="left", padx=4)
        ttk.Button(btns, text="View initial", command=self.view_initial).pack(side="left", padx=4)

        # ---- results table ----
        res = ttk.LabelFrame(frm, text="Results", padding=6)
        res.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        self.res_rows = {}
        for i, m in enumerate(("opening", "fpfh", "collapse")):
            ttk.Label(res, text=m, width=10).grid(row=i, column=0, sticky="w")
            lbl = ttk.Label(res, text="—", width=40)
            lbl.grid(row=i, column=1, sticky="w")
            vb = ttk.Button(res, text="View", state="disabled",
                            command=lambda mm=m: self.view_result(mm))
            vb.grid(row=i, column=2, padx=4)
            sb = ttk.Button(res, text="Save merged", state="disabled",
                            command=lambda mm=m: self.save_result(mm))
            sb.grid(row=i, column=3, padx=4)
            self.res_rows[m] = (lbl, vb, sb)

        # ---- log ----
        ttk.Label(frm, text="Log").grid(row=5, column=0, sticky="w", **pad)
        self.log = tk.Text(frm, height=14, width=92, wrap="none")
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(6, weight=1)
        frm.columnconfigure(1, weight=1)

        self.root.after(100, self._drain)

    # ---- helpers ----
    def _browse(self, var):
        p = filedialog.askopenfilename(
            title="Select PLY",
            filetypes=[("PLY", "*.ply"), ("All", "*.*")])
        if p:
            var.set(p)

    def _logln(self, msg):
        self.q.put(msg)

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "__done__":
                    self._on_done(msg[1])
                else:
                    self.log.insert("end", msg + "\n")
                    self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _vox(self):
        v = self.voxel.get().strip().lower()
        return None if v in ("", "auto") else float(v)

    # ---- load ----
    def load(self):
        if self.busy:
            return
        a, b = self.pathA.get().strip(), self.pathB.get().strip()
        if not (os.path.isfile(a) and os.path.isfile(b)):
            messagebox.showerror("Missing files", "Pick both Mesh A and Mesh B PLYs.")
            return
        n = int(self.nsamp.get())
        self.busy = True
        self._logln(f"Loading A: {os.path.basename(a)}")
        self._logln(f"Loading B: {os.path.basename(b)}")

        def work():
            try:
                self.A = align.load_mesh(a, n_sample=n)
                self.B = align.load_mesh(b, n_sample=n)
                self._logln(f"  A extent={self.A.extent:.4g}  d_open={np.round(self.A.d_open,3)}")
                self._logln(f"  B extent={self.B.extent:.4g}  d_open={np.round(self.B.d_open,3)}")
                self._logln("Loaded. Ready to run / view initial.")
            except Exception as e:
                self._logln(f"ERROR: {e}")
            self.q.put(("__done__", None))

        threading.Thread(target=work, daemon=True).start()

    # ---- run ----
    def run(self, which):
        if self.busy:
            return
        if self.A is None or self.B is None:
            messagebox.showinfo("Load first", "Click 'Load meshes' first.")
            return
        self.busy = True
        vox = self._vox()
        methods = ["opening", "fpfh", "collapse"] if which == "all" else [which]
        fns = {"opening": align.align_opening, "fpfh": align.align_fpfh,
               "collapse": align.align_collapse}

        def work():
            for m in methods:
                self._logln(f"\n=== {m} ===")
                try:
                    fn = fns[m]
                    r = fn(self.A, self.B, voxel=vox, log=self._logln)
                    self.results[m] = r
                    self.q.put(("__done__", m))
                except Exception as e:
                    self._logln(f"ERROR ({m}): {e}")
            self.q.put(("__done__", None))

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, method):
        if method in self.results:
            r = self.results[method]
            lbl, vb, sb = self.res_rows[method]
            lbl.config(text=f"closed={r.closed:.3f}  detail={r.detail:.3f}  fit={r.fitness:.3f}  "
                            f"(closed→1 right side; detail→1 right spin)")
            vb.config(state="normal")
            sb.config(state="normal")
        self.busy = False

    # ---- visualization (main thread) ----
    def view_initial(self):
        if self.A is None or self.B is None:
            messagebox.showinfo("Load first", "Click 'Load meshes' first.")
            return
        o3d.visualization.draw_geometries(
            align.view_geometries(self.A, self.B, T=None),
            window_name="Initial (unaligned)  green=A  red=B")

    def view_result(self, method):
        r = self.results.get(method)
        if r is None:
            return
        o3d.visualization.draw_geometries(
            align.view_geometries(self.A, self.B, T=r.transform),
            window_name=f"{method}  fitness={r.fitness:.3f}  green=A  red=B")

    def save_result(self, method):
        r = self.results.get(method)
        if r is None:
            return
        p = filedialog.asksaveasfilename(
            title="Save merged PLY", defaultextension=".ply",
            initialfile=f"merged_{method}.ply",
            filetypes=[("PLY", "*.ply")])
        if p:
            align.save_merged(self.A, self.B, r.transform, p)
            self._logln(f"Saved merged ({method}) -> {p}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
