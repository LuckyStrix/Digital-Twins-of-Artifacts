from pipeline.parsers import AlignParser, ColmapParser, PhotosParser, ReconParser


def feed_all(parser, lines):
    return [r for r in map(parser.feed, lines) if r]


def test_photos_parser():
    p = PhotosParser()
    assert p.feed("Processing 40 images with birefnet-general") == (2, "Processing 40 images…")
    assert p.feed("[1/40] IMG_0001.JPG") == (7, "Image 1/40")
    assert p.feed("[40/40] IMG_0040.JPG") == (95, "Image 40/40")
    assert p.feed("[done] wrote 40 images") == (100, "Complete")
    assert p.feed("unrelated") is None


def test_colmap_parser_sparse_and_dense_single_component():
    p = ColmapParser()
    out = feed_all(p, [
        "[step] colmap feature_extractor --database_path db",
        "Processing file [5/10]",
        "[step] colmap exhaustive_matcher",
        "[step] colmap mapper",
        "sparse models found: 1",
        "[dense] colmap image_undistorter",
        "[dense] colmap patch_match_stereo",
        "Processing view 1 / 4",
        "Processing view 4 / 4",
        "Processing view 1 / 4",   # second (geometric) sweep
        "Processing view 4 / 4",
        "[dense] colmap stereo_fusion",
        "Processing view 2 / 4",
        "dense cloud output: fused.ply",
    ])
    assert out == [
        (5, "Feature extraction"),
        (12, "Feature extraction: 5/10"),
        (20, "Feature matching"),
        (38, "Sparse mapping"),
        (40, "Image undistortion"),
        (55, "Patch match stereo"),
        # view 1/4 maps to 55.9% -> 55, not past the current value: no update
        (58, "Photometric: view 4/4"),
        (59, "Geometric: view 1/4"),
        (62, "Geometric: view 4/4"),
        (82, "Stereo fusion"),
        (91, "Fusion: view 2/4"),
        (100, "Dense cloud complete"),
    ]


def test_colmap_parser_components_and_range():
    p = ColmapParser(50, 100)
    out = feed_all(p, [
        "sparse models found: 2",
        "[dense component 0] colmap image_undistorter",
        "[dense component 1] colmap image_undistorter",
        "dense cloud output: x",
    ])
    assert out == [
        (70, "Image undistortion (part 1/2)"),
        (85, "Image undistortion (part 2/2)"),
        (100, "Dense cloud complete"),
    ]


def test_colmap_parser_never_goes_backwards():
    p = ColmapParser()
    assert p.feed("[step] colmap mapper") == (38, "Sparse mapping")
    assert p.feed("[step] colmap feature_extractor") is None


def test_align_parser_order_matches_real_output():
    # Real run.py order: FPFH ... Best method, then ICP refine, then seam, then save.
    p = AlignParser()
    out = feed_all(p, [
        "=== fpfh ===",
        "FPFH on 60000 points",
        "RANSAC fitness=0.8",
        "refine fitness=0.9",
        "chosen fpfh",
        "Best method: fpfh",
        "=== refine ===",
        "[icp] stage 1/4 dist=3",
        "[icp] stage 2/4 dist=1",
        "[icp] stage 3/4 dist=0.5",
        "[icp] stage 4/4 dist=0.25",
        "[icp] after: fitness=0.95",
        "=== resolve seam ===",
        "Saved merged cloud to merged_fpfh.ply",
    ])
    assert [pct for pct, _ in out] == [10, 20, 40, 60, 85, 90, 91, 92, 94, 96, 97, 98, 99, 100]
    assert out[-2] == (99, "Resolving seam…")


def test_align_parser_seam_without_refine():
    p = AlignParser()
    feed_all(p, ["Best method: fpfh"])
    assert p.feed("=== resolve seam ===") == (99, "Resolving seam…")


def test_recon_parser():
    p = ReconParser()
    out = feed_all(p, [
        "[step] input",
        "[step] poisson reconstruction",
        "[step] input",  # repeated / earlier steps don't regress
        "[step] simplified export",
        "gltf output: model.gltf",
    ])
    assert out == [
        (7, "Loading input…"),
        (57, "Poisson reconstruction…"),
        (97, "Simplified web-viewer export…"),
        (100, "GLTF written"),
    ]
