"""Progress parsers for the pipeline subprocesses.

Stateful; call ``feed(line)`` for every output line and get back
``(pct, text)`` when the line advances progress, else ``None``.
"""

import re


class PhotosParser:
    def __init__(self):
        self._total = 0
        self._note = ""

    def feed(self, line: str):
        if line.startswith("[warn]") and "running on the CPU" in line:
            self._note = " (on CPU - slow)"
            return (2, "rembg fell back to the CPU - slow")
        m = re.search(r'Processing (\d+) image', line)
        if m:
            self._total = int(m.group(1))
            return (2, f"Processing {self._total} images…")
        m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            if n > 0:
                self._total = n
                return (int(5 + 90 * i / n), f"Image {i}/{n}{self._note}")
        if re.search(r'\[done\]', line, re.IGNORECASE):
            return (100, "Complete")
        return None


class ColmapParser:
    # Sparse steps fire once, before any per-component dense work.
    # (pattern, frac_at_start, step_id, label)
    _SPARSE_STEPS = [
        (r'\[step\] colmap feature_extractor',
         0.05, 'extract', "Feature extraction"),
        (r'\[step\] colmap (?:exhaustive_matcher|vocab_tree_matcher|sequential_matcher)',
         0.20, 'match', "Feature matching"),
        (r'\[step\] colmap mapper',
         0.38, 'map', "Sparse mapping"),
    ]

    # Dense work (undistort -> patch match stereo -> fusion) runs once per
    # sparse-model component and occupies the [_DENSE_BASE, 1.0) fraction,
    # split evenly across however many components run_colmap_mvs.py found.
    # Within one component's share, these are the sub-step offsets (as a
    # fraction of that component's own [0, 1) range), mirroring the ratios
    # the flat single-pass version used to use.
    _DENSE_BASE     = 0.40
    _COMP_UNDISTORT = 0.25
    _COMP_PMS       = 0.367
    _COMP_FUSION    = 0.70
    _COMP_END       = 1.0

    _RE_N_COMPONENTS = re.compile(r'sparse models found:\s*(\d+)')
    _RE_UNDISTORT = re.compile(r'\[.*?(?:component (\d+))?\] colmap image_undistorter')
    _RE_PMS       = re.compile(r'\[.*?(?:component (\d+))?\] colmap patch_match_stereo')
    _RE_FUSION    = re.compile(r'\[.*?(?:component (\d+))?\] colmap stereo_fusion')
    _RE_DONE      = re.compile(r'dense cloud output:')
    _RE_VIEW      = re.compile(r'Processing view\s+(\d+)\s*/\s*(\d+)')
    _RE_FILE      = re.compile(r'Processing file \[(\d+)/(\d+)\]')

    def __init__(self, lo: int = 0, hi: int = 100):
        self._lo   = lo
        self._hi   = hi
        self._pct  = lo
        self._step = None
        self._n_components = 1
        self._comp_idx = 0
        # patch_match_stereo (with the default geom_consistency=true) makes
        # a photometric-only sweep over all views, then a second sweep that
        # refines using geometric consistency -- both print the same
        # "Processing view i/n" line. Track sweep boundaries via the view
        # counter resetting, so each sweep gets its own slice + label
        # instead of being lumped together as "Photometric".
        self._pms_pass    = 0
        self._pms_last_i  = 0

    def _scale(self, frac: float) -> int:
        return int(self._lo + (self._hi - self._lo) * frac)

    def _emit(self, frac: float, label: str):
        pct = self._scale(frac)
        if pct > self._pct:
            self._pct = pct
            return (pct, label)
        return None

    def _component_frac(self, offset: float) -> float:
        """Map an offset within one component's [0, 1) range to a global frac."""
        width = (1.0 - self._DENSE_BASE) / self._n_components
        return self._DENSE_BASE + width * (self._comp_idx + offset)

    def _part_suffix(self) -> str:
        return f" (part {self._comp_idx + 1}/{self._n_components})" if self._n_components > 1 else ""

    def feed(self, line: str):
        m = self._RE_N_COMPONENTS.search(line)
        if m:
            self._n_components = max(1, int(m.group(1)))
            return None

        for pat, frac, step_id, label in self._SPARSE_STEPS:
            if re.search(pat, line):
                self._step = step_id
                return self._emit(frac, label)

        m = self._RE_UNDISTORT.search(line)
        if m:
            self._step = 'undistort'
            if m.group(1) is not None:
                self._comp_idx = int(m.group(1))
                self._n_components = max(self._n_components, self._comp_idx + 1)
            return self._emit(self._component_frac(0.0), f"Image undistortion{self._part_suffix()}")

        m = self._RE_PMS.search(line)
        if m:
            self._step = 'pms'
            if m.group(1) is not None:
                self._comp_idx = int(m.group(1))
                self._n_components = max(self._n_components, self._comp_idx + 1)
            self._pms_pass   = 0
            self._pms_last_i = 0
            return self._emit(self._component_frac(self._COMP_UNDISTORT), f"Patch match stereo{self._part_suffix()}")

        m = self._RE_FUSION.search(line)
        if m:
            self._step = 'fusion'
            if m.group(1) is not None:
                self._comp_idx = int(m.group(1))
                self._n_components = max(self._n_components, self._comp_idx + 1)
            return self._emit(self._component_frac(self._COMP_FUSION), f"Stereo fusion{self._part_suffix()}")

        if self._RE_DONE.search(line):
            self._step = 'done'
            self._pct = self._hi
            return (self._hi, "Dense cloud complete")

        # "Processing view X/N" inside patch_match_stereo: alternates between
        # a photometric sweep and a geometric-consistency sweep.
        if self._step == 'pms':
            m = self._RE_VIEW.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n > 0:
                    if i < self._pms_last_i:
                        self._pms_pass += 1
                    elif self._pms_pass == 0:
                        self._pms_pass = 1
                    self._pms_last_i = i

                    pass_start = self._COMP_UNDISTORT
                    pass_width = self._COMP_PMS - self._COMP_UNDISTORT
                    sub_idx    = min(self._pms_pass - 1, 1)  # cap at 2 sweeps' worth of range
                    sub_start  = pass_start + pass_width * (sub_idx / 2)
                    sub_end    = pass_start + pass_width * ((sub_idx + 1) / 2)
                    offset     = sub_start + (sub_end - sub_start) * i / n
                    kind = "Photometric" if self._pms_pass % 2 == 1 else "Geometric"
                    return self._emit(self._component_frac(offset), f"{kind}: view {i}/{n}{self._part_suffix()}")

        # "Processing view X/N" inside stereo_fusion
        elif self._step == 'fusion':
            m = self._RE_VIEW.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n > 0:
                    offset = self._COMP_FUSION + (self._COMP_END - self._COMP_FUSION) * i / n
                    return self._emit(self._component_frac(offset), f"Fusion: view {i}/{n}{self._part_suffix()}")

        # "Processing file [N/M]" — feature extraction per-image
        if self._step == 'extract':
            m = self._RE_FILE.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n > 0:
                    frac = 0.05 + (0.20 - 0.05) * i / n
                    return self._emit(frac, f"Feature extraction: {i}/{n}")

        return None


class AlignParser:
    _STEPS = [
        (r'=== (?:fpfh|opening|collapse) ===', 10, "Initialising alignment…"),
        (r'FPFH on',                            20, "Computing FPFH features…"),
        (r'RANSAC fitness=',                    40, "RANSAC global registration…"),
        (r'refine ',                            60, "Refining alignment (ICP)…"),
        (r'chosen ',                            85, "Selecting best candidate…"),
        (r'Best method:',                       90, "Evaluating methods…"),
        (r'=== refine ===',                     91, "ICP refinement…"),
        # Seam resolution runs after ICP refinement ("[icp] after" = 98), so it
        # must sit above it or the label never updates.
        (r'=== resolve seam ===',               99, "Resolving seam…"),
        (r'\[icp\] stage 1/',                   92, "ICP stage 1…"),
        (r'\[icp\] stage 2/',                   94, "ICP stage 2…"),
        (r'\[icp\] stage 3/',                   96, "ICP stage 3…"),
        (r'\[icp\] stage [4-9]/',               97, "ICP final stage…"),
        (r'\[icp\] after',                      98, "ICP finished…"),
        (r'Saved merged',                      100, "Merged cloud saved"),
    ]

    def __init__(self):
        self._pct = 0

    def feed(self, line: str):
        for pattern, pct, text in self._STEPS:
            if re.search(pattern, line):
                if pct > self._pct:
                    self._pct = pct
                    return (pct, text)
        return None


class ReconParser:
    _STEPS = [
        (r'\[step\] random downsample',           4,  "Random downsample…"),
        (r'\[step\] input',                        7,  "Loading input…"),
        (r'\[step\] voxel downsample',            11,  "Voxel downsample…"),
        (r'\[step\] statistical outlier removal',  17,  "Statistical outlier removal…"),
        (r'\[step\] radius outlier removal',       22,  "Radius outlier removal…"),
        (r'\[step\] bbox quantile crop',           26,  "Bounding box crop…"),
        (r'\[step\] cluster cleanup',              32,  "DBSCAN cluster cleanup…"),
        (r'\[step\] optional mirror',              35,  "Mirror check…"),
        (r'\[step\] estimate \+ orient normals',   42,  "Estimating normals…"),
        (r'\[step\] write cleaned cloud',          50,  "Writing cleaned cloud…"),
        (r'\[step\] poisson reconstruction',       57,  "Poisson reconstruction…"),
        (r'\[step\] density trim',                 67,  "Density trim…"),
        (r'\[step\] mesh cleanup',                 73,  "Mesh cleanup…"),
        (r'\[step\] fill mesh holes',               76,  "Filling holes…"),
        (r'\[step\] mesh normal orientation',      79,  "Normal orientation check…"),
        (r'\[step\] normalize pose',                82,  "Centering & flattening pose…"),
        (r'\[step\] optional decimation',          85,  "Decimation…"),
        (r'\[step\] textured output',              90,  "Textured output…"),
        (r'\[step\] write mesh exports',           94,  "Writing mesh exports…"),
        (r'\[step\] simplified export',            97,  "Simplified web-viewer export…"),
        (r'gltf output:',                         100,  "GLTF written"),
    ]

    def __init__(self):
        self._pct = 0

    def feed(self, line: str):
        for pattern, pct, text in self._STEPS:
            if re.search(pattern, line):
                if pct > self._pct:
                    self._pct = pct
                    return (pct, text)
        return None
