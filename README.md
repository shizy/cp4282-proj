# Exploring 3DGS Compression with Image Codecs
This repository holds a single notebook, `gs_image_codec.ipynb`. This notebook represents all the experiments conducted to explore 3DGS compression by packing its splats into a 2D grid and storing each attribute as standard image files (PNG, WebP, JPEG, AVIF). Everything runs after training: there is no retraining or fine-tuning.

On the NeRF-Synthetic Lego scene, the best configuration shrinks the model from 80.0 MB to 6.2 MB (12.9× smaller than the float `.ply`, 2.5× smaller than a 1D LZMA baseline), losing 0.47 dB of test PSNR.

## Repository contents

| Item | Where |
| --- | --- |
| `README.md` | `main` branch |
| `gs_image_codec.ipynb` | `main` branch |
| Scene data (scene checkpoints and test views used, about 500 MB) | `cp4282_data.zip` under [Releases](https://github.com/shizy/cp4282-proj/releases/tag/v1.0.0) |

## Quick start (Google Colab, recommended)

The notebook was developed and run on Colab with a Tesla T4 GPU.

1. **Open the notebook in Colab.** Go to [colab.research.google.com](https://colab.research.google.com), choose *File → Open notebook → GitHub*, and paste this repository's URL.
2. **Select a GPU runtime.** *Runtime → Change runtime type → T4 GPU*. On a CPU runtime the notebook still runs, but at reduced resolution and with fewer views (see [Running without a GPU](#running-without-a-gpu)).
3. **Download the data.** Add a new cell at the top of the notebook and run:

   ```bash
   !wget -q https://github.com/shizy/cp4282-proj/releases/download/v1.0.0/cp4282_data.zip
   !unzip -q -o cp4282_data.zip -d data
   ```

   After unzipping, the files must match the layout in [Data layout](#data-layout).
4. **Run everything.** *Runtime → Run all*. The first cell installs the dependencies. The first GPU render also compiles the Warp kernels, which takes about 10–15 seconds once.

## Data layout

The notebook looks for each scene at these paths, relative to the notebook's working directory:

```
data/
└── nerf_synthetic/
    ├── lego/
    │   ├── lego.ply                 # trained 3DGS checkpoint (INRIA format)
    │   ├── transforms_test.json     # test camera poses
    │   └── test/                    # test images (r_0.png ... r_199.png)
    ├── materials/
    │   ├── materials.ply
    │   ├── transforms_test.json
    │   └── test/
    └── ship/
        ├── ship.ply
        ├── transforms_test.json
        └── test/
```

To use a different location, edit `ply_path` and `data_dir` in the configuration cell (§1).

If the files for a scene are missing, the notebook does not stop. It silently builds a small **synthetic scene** instead, so every cell still runs. The scene-loading cell (§4) then prints `[SYNTHETIC FALLBACK]`. If you see that message, the data was not found, and the numbers will not match the report.

## Running locally

You need Python 3.10 or newer and, for full-resolution results, an NVIDIA GPU with CUDA.

```bash
pip install warp-lang plyfile zstandard "pillow>=11.3" scikit-image numpy pandas matplotlib
pip install torch lpips                                    # LPIPS metric
pip install git+https://github.com/fraunhoferhhi/PLAS.git  # PLAS ordering
```

Pillow 11.3 or newer is needed for AVIF support. The notebook's first cell only installs the core packages automatically on Colab, but it always attempts to install LPIPS and PLAS.

Then unzip the release into the repository folder so that `data/nerf_synthetic/...` sits next to the notebook, and run all cells.

## Configuration

All settings are in the `CFG` dictionary in §1. The main ones:

| Setting | Default | Meaning |
| --- | --- | --- |
| `scene` | `"lego"` | Scene used for all sweeps (§4–§15) |
| `scenes` | `["lego", "materials", "ship"]` | Scenes evaluated in the cross-scene run (§16) |
| `res_scale` | `1.0` on GPU, `0.25` on CPU | Render resolution relative to 800 × 800 |
| `n_sweep_views` | `40` on GPU, `8` on CPU | Test views used inside the sweeps |
| `n_eval_views` | all 200 on GPU, `25` on CPU | Test views for final evaluation |
| `prune_keep` | `None` | Fraction of splats to keep with importance pruning. `None` disables pruning (the report uses unpruned checkpoints) |
| `psnr_budget_db` | `0.5` | PSNR budget for the hybrid search |
| `out_dir` | `"out"` | Where encoded images and outputs are written |

## What the notebook does

| Section | Content |
| --- | --- |
| §1–§3 | Configuration, `.ply` reading and writing, camera loading |
| §2b | Forward tile-based 3DGS rasterizer written in NVIDIA Warp |
| §4 | Load the scene and render the float model |
| §5 | Optional importance pruning (off by default) |
| §6 | Fixed quantizer, rotation–scale canonical forms, and the 1D LZMA/zlib/zstd baseline |
| §7 | Orderings: unsorted, Morton, iterative swap, PLAS |
| §8 | Neighbour alignment of the 48 equivalent rotation–scale forms |
| §9 | Image layout, codecs, and the encode/decode container |
| §10–§11 | Experiment A: lossless ordering × alignment ablation, plus a bit-exact round-trip check |
| §12 | Ordering and alignment under lossy AVIF coding |
| §13 | Experiment B: per-attribute lossy sensitivity and JPEG vs AVIF comparison |
| §14 | Experiment C: best hybrid configuration within the PSNR budget, with rendered comparisons |
| §15 | Experiment D: sorting, encoding and decoding time against splat count |
| §16 | Condensed pipeline on Lego, Materials and Ship, plus Ship renders |

Cells depend on earlier ones, so run them in order.

## Expected results

With the release data on a T4 GPU, the final cell (§16) should give approximately:

| Scene | 1D LZMA (MB) | Hybrid (MB) | Smaller | ΔPSNR (dB) |
| --- | --- | --- | --- | --- |
| Lego | 15.50 | 6.20 | 2.50× | −0.47 |
| Materials | 13.78 | 4.79 | 2.87× | −0.18 |
| Ship | 15.93 | 5.90 | 2.70× | −0.38 |

The swap and PLAS orderings are randomized, so sizes can differ slightly between runs (for example 14.91 vs 14.92 MB for the best lossless Lego grid). Timing numbers depend on the GPU.

## Running without a GPU

On a CPU runtime the notebook switches automatically to 200 × 200 renders, 8 sweep views, and 25 evaluation views. It runs end to end but much more slowly, and the PSNR values will not match the report.
