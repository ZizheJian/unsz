# UNSZ

English | [中文](README_CN.md)

UNSZ is a parallel, prediction-based, error-bounded lossy compressor for unstructured scientific data, that is, floating-point values sampled at irregularly distributed spatial coordinates. It needs only the coordinates of the points and does not require mesh connectivity. The compressor and the decompressor build the same prediction structures from the coordinates, so these structures are not stored in the compressed stream. The prediction residuals are quantized, Huffman-coded, and compressed with Zstd.

A dataset may contain several fields that share the same coordinates, and each field may contain several slices, such as one per time step or vertical level.

## Features

- **Pointwise error bounds.** Every reconstructed valid value satisfies an absolute bound, a bound relative to the value range of its slice, or a bound relative to the value range of its whole field.
- **In-slice prediction.** Each value is predicted from nearby points in the same slice. UNSZ rescales the coordinate axes when this helps, builds the prediction sequence with a reverse construction, weights the references with radial basis function (RBF) interpolation, and tightens the error bounds of the points predicted first.
- **Cross-slice prediction.** For fields with multiple slices, UNSZ also predicts each value from neighboring slices and selects the best pipeline for each field automatically.
- **Exceptional values.** Fill values and other exceptional values are excluded from prediction and encoded compactly as labels.
- **Parallelism.** Fields are distributed over workers, and each slice can be split into spatial partitions that are compressed in parallel with OpenMP.

## Paper

*UNSZ: A Parallel Error-Bounded Lossy Compressor for Unstructured Scientific Data* (under review).

The paper describes the algorithms and the experimental results. This README explains how to build and run UNSZ, how to reproduce the experiments, and how the code is organized.

## 1. Build

Building requires CMake, a C++17 compiler with AVX2 support, and Python 3. The C++ dependencies (Zstd and an argument parsing library) and all baseline compressors are included as git submodules.

```bash
git clone --recurse-submodules https://github.com/ZizheJian/unsz.git
cd unsz
cmake -S c_codes -B build -DCMAKE_INSTALL_PREFIX="$PWD/install" -DCODE_NAME=s1_compress
cmake --build build --parallel 8
cmake --install build
```

For an existing clone, run `git submodule update --init --recursive` first. The build produces `install/bin/s1_compress` and `install/bin/s1_decompress`.

The Python environment is used to run batch experiments, process results, and plot figures:

```bash
python3 -m venv unsz_env
source unsz_env/bin/activate
python -m pip install numpy scipy matplotlib pyvista
python -m pip install -e third_party/paperfig
```

## 2. Usage

### 2.1. Input format

UNSZ reads float32 coordinates and float32 values. A data directory contains at least the following files:

| File | Content |
|---|---|
| `coords.dat` | Coordinates stored point by point, `dim` float32 values per point |
| `<field>.dat.<slice>` | One slice of float32 values, one value per point, in the same point order as `coords.dat` |

For example, a three-dimensional temperature field with 31 slices:

```text
Ocean/
├── coords.dat
├── temperature.dat.0
├── ...
└── temperature.dat.30
```

### 2.2. Compression and decompression

```bash
install/bin/s1_compress -data-dir /path/to/Ocean -field temperature -slice '[0,30]' -M REL 1e-3
install/bin/s1_decompress -data-dir /path/to/Ocean -field temperature -slice '[0,30]' -M REL 1e-3
```

The compressed streams are written as `<field>.dat.<slice>.unsz`, and the decompressed data as `<field>.dat.<slice>.unsz.bin`. Both go to the data directory unless `-output-dir` is given. The programs print the total bit rate, the PSNR, the ratio of the maximum error to the error bound, and the preprocessing and codec times of both sides. The compressed stream does not store the error bound, so the decompressor must receive the same `-M` option or label configuration as the compressor.

| Option | Meaning |
|---|---|
| `-data-dir`, `-output-dir` | Data directory, and output directory (default: the data directory) |
| `-field`, `-field-list` | One field, or several fields that share the coordinates, written as `{a,b,c}` |
| `-slice` | Closed range `[a,b]`, half-open range `[a,b)`, or list `{a1,a2,...}`; quoting is recommended |
| `-slice-coordinate-file` | Actual coordinates of unevenly spaced slices, as a raw float64 array; must be identical on both sides |
| `-tensor-metadata` | Field definitions for slices on two tensor axes (Section 2.4); cannot be combined with `-slice` |
| `-coordinate-dim` | Number of coordinate dimensions (default: 3) |
| `-M <REL or ABS> <eb>` | Error bound for data without exceptional values; use either this or `-label-config` |
| `-label-config` | Label configuration in JSON (Section 2.3) |
| `-partition-num`, `-thread-num` | Split the points into a power-of-two number of independent spatial partitions and process them with OpenMP threads |
| `-field-worker-num`, `-field-worker-map` | Distribute several fields over several workers (Section 2.5) |
| `-value-pipeline`, `-label-pipeline`, `-coordinate-rescaling` | Override the automatically selected cross-slice pipelines and coordinate rescaling, for diagnosis and ablation only. Value pipelines: `independent`, `extrapolation`, `interpolation`, `anchor_interpolation`, where the last three accept the suffix `_spatial_residual`. Label pipelines: `independent`, `extrapolation`, `interpolation_spatial_residual`, `anchor_interpolation`. Coordinate rescaling: `off`, `on`, or `auto` |
| `-compression-stage-time`, `-decompression-stage-time` | Print the time of every stage |
| `-skip-decompressed-output` | Verify the pointwise errors during decompression without writing the decompressed files |

### 2.3. Label configuration

Without `-label-config`, all values are valid and `-M` gives the error bound. When the data contain exceptional values, such as fill values, a label configuration divides the points into valid values (label 0) and exceptional values (label 1). For example, for Katrina:

```json
{
  "version":3,
  "labels":[
    {"method":"predict","eb_mode":"REL","eb_value":0.01,"condition":{"operator":">=","value":-1000}},
    {"method":"constant"}
  ]
}
```

The index in `labels` is the label number. UNSZ tests the `condition` of every label except the last one in order. The first condition that holds determines the label, and a point that satisfies none of them receives the last label, so the last label has no `condition`. The operators are `<`, `<=`, `>`, `>=`, `==`, and `!=`. The `method` is either `predict` (RBF prediction) or `constant` (constant encoding). A label with `method` `predict` must give `eb_mode` and `eb_value`. With `ABS`, `eb_value` is the absolute error bound. With `REL`, it is multiplied by the value range of this label in the slice, and with `FIELD_REL`, by the value range of this label in the whole field. If two labels are configured but a field contains only one of them, UNSZ automatically compresses it with a single label.

### 2.4. Fields with two tensor axes

Data shaped as `[time,height,point]`, such as Ocean-large, are described with `-tensor-metadata`. The slice with index `i` on the first axis and index `j` on the second axis is stored in `<field>.dat.<i*shape[1]+j>`. One field list may mix different shapes:

```json
{
  "version":1,
  "fields":{
    "timeMonthly_avg_velocityMeridional":{
      "shape":[12,80],
      "axis0_coordinates":"time_12.f64",
      "axis1_coordinates":"height_center_80.f64",
      "active_level_count_file":"active_level_count.u32",
      "active_level_count_offset":0
    }
  }
}
```

The coordinate files are strictly increasing float64 arrays. The active level count file holds one uint32 per horizontal point, with offset 0 for level centers and offset 1 for level interfaces. When active levels are configured, points with `height_index>=maxLevelCell+offset`, non-finite values, and `-99999` all receive label 1 (exceptional values).

### 2.5. Multiple fields and two-level parallelism

Several fields that share the coordinates can be compressed in one process with `-field-list`, and all fields without coordinate rescaling share the same in-slice DAG. Adding `-field-worker-num N` and `-field-worker-map {w0,w1,...}`, which has one entry per field, distributes the fields over N workers. Each worker still parallelizes its slices with `-partition-num` and `-thread-num`. The worker assignment for Ocean-large is computed by `benchmarks/common.py`.

## 3. Reproducing the experiments

The datasets are not included in this repository. The scripts in `py_codes/prepare_datasets/` convert each dataset from its original format into the input format of Section 2.1.

### 3.1. Moderate-size datasets

Each JSON file in `configs/` describes the experiments on one dataset. `t1_start_c_code.py` reads it, builds and runs UNSZ and UMC, and writes the results to CSV files:

```bash
./unsz_env/bin/python t1_start_c_code.py katrina
```

The top level of the JSON file is a task list. Every task needs a unique `name`, an absolute `data_dir`, and a result path `csv_path` (with the placeholder `{field}` for multiple fields). `field_list` and `compressor_list` are enumerated exhaustively. Tasks without exceptional values list their error bounds in `rel_eb_list`, and tasks with exceptional values list complete label configurations in `unsz.label_config_path_list`. UMC runs once for every REL error bound that appears in these label configurations and serves as the baseline for comparisons at equal PSNR.

```json
[
  {
    "name":"les_velocity",
    "data_dir":"/path/to/LES",
    "csv_path":"results/current/fields/les_{field}.csv",
    "field_list":["velocity_x","velocity_y"],
    "rel_eb_list":[0.1,0.01,0.001,0.0001,0.00001],
    "compressor_list":["unsz","umc"],
    "slice_num":1,
    "coordinate_dim":3,
    "unsz":{},
    "umc":{"from_edge":0,"reorder":1,"method":"NBP-IDW","neighbor_num":4}
  }
]
```

| Script | Purpose |
|---|---|
| `t1_start_c_code.py` | Entry point of the experiments; builds incrementally by default, add `--skip-build` to reuse an existing installation |
| `py_codes/run_configurations.slurm`, `py_codes/run_spatial_field_benchmark.slurm` | Run one or more JSON files from `configs/` on a compute node |
| `py_codes/parse_unsz_output.py`, `py_codes/parse_umc_output.py` | Parse the output of the compressors |
| `py_codes/summarize_current_results.py` | Compute the compression ratio and time ratio of UNSZ over UMC for every field and write `results/current/summary.csv` |
| `py_codes/plot_current_benchmarks.py` | Plot the rate–distortion and time figures of the moderate-size datasets |
| `py_codes/prepare_datasets/` | Convert the datasets into the input format of Section 2.1 |
| `t3_visualize.py` | Visualize Katrina and Ocean fields before and after coordinate rescaling |

UMC uses the official code in `third_party/umc`. `third_party/build_umc.sh` replaces its timing files before building it, and the experiments always use `reorder=1`. The mesh of ADCIRC-Haiyan has two connected components, and the reordering of UMC traverses only one of them, so UMC is reported as unable to run on it.

### 3.2. Ocean-large

The experiments on Ocean-large are in `benchmarks/`, and the README in each directory explains how to run them:

| Directory | Content |
|---|---|
| `benchmarks/two_level/` | UNSZ with field workers and spatial partitions |
| `benchmarks/umc_field_parallel/` | UMC parallelized over fields, with serial UMC in every worker |
| `benchmarks/slice_parallel/` | SZ3-1D and PFPL parallelized over slices |
| `benchmarks/field_relative/` | UNSZ, SZ3-ND, and QoZ-ND under per-field error bounds |
| `benchmarks/scaling/` | Scalability with different numbers of cores |
| `benchmarks/tensor_1d/` | Building and single-thread runs of the one-dimensional baselines |

`py_codes/plot_ocean_large.py` reads these results and plots the Ocean-large figures of the paper.

## 4. Code structure

The C++ code is in `c_codes/`. `s1_compress.cpp` and `compress_cli.cpp` are the entry points of the compressor, and `s1_decompress.cpp` is the entry point of the decompressor. Both share the command-line parsing and field-worker scheduling in `cli_common.hpp`. The core implementation is in `c_codes/unsz_core/`. `unsz.hpp` defines all data structures and begins with notes that relate the terms in the code to those in the paper.

| File | Content |
|---|---|
| `compress_entry.tpp` | Overall compression flow: preprocessing per field, then compression per slice |
| `coordinate_rescaling.tpp` | Learning the coordinate rescaling and deciding whether to apply it |
| `tree.tpp`, `partition.tpp` | kd-tree and spatial partitions |
| `generate_prediction_structure_and_weights.tpp` | Reverse construction of the prediction sequence and the references |
| `prediction_weights.tpp` | Batched solution of the RBF weights |
| `eb_schedule.tpp` | Error bound of each point along the prediction sequence |
| `automatic_pipeline_selection.tpp` | Automatic selection of the cross-slice pipeline |
| `prediction_framework.tpp` | Prediction method of each slice and its dispatch to template parameters |
| `init.tpp`, `tensor_dependency.tpp`, `tensor_prediction.tpp` | Order of the slices, cross-slice references, and cross-slice prediction |
| `label_config.tpp` | Label configuration and conversion of error bounds |
| `compress.tpp`, `decompress.tpp` | Pointwise prediction, quantization, and reconstruction |
| `spatial_prediction.tpp` | In-slice prediction with RBF weights |
| `compression_format.tpp` | Encoding, assembling, and parsing the compressed stream |

### Regression test

After changing the C++ code, a byte-exact regression test checks that the output of the default path is unchanged. It compresses and decompresses 15 fixed cases with a reference binary and a new binary and compares the SHA256 of all compressed streams and decompressed files. The output files go to `$SCRATCH/unsz_regression` by default, and `--work-dir` changes this location:

```bash
./unsz_env/bin/python tests/regression/run_regression.py --binary-dir build/regression_new/install/bin --manifest results/regression/candidate.json --compare results/regression/reference.json
```

`results/regression/reference.json` records the reference output from before the code cleanup.
