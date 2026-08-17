# AI-Assisted Development Log

## Initial Project Structure

- Prompt: Create the initial project structure for the UQ COMP3710 Part 3 Sierpiński carpet project, including only `AGENTS.md`, `README.md`, `prompts.md`, `.gitignore`, `src/`, `jobs/`, and `outputs/.gitkeep`, without implementing the fractal algorithm.
- Assistance summary: Created the requested scaffold and project documentation rules.
- Issues encountered: None.
- Changes made: Added project rule documentation, a minimal README, this prompt log, Git ignore rules, and empty source/job/output directories.

## Stage 1 Baseline Renderer

- Prompt: Implement Stage 1 of the UQ COMP3710 Part 3 project: a baseline Sierpiński carpet using PyTorch. Create `src/sierpinski.py` and `jobs/render_job.sh`; do not implement fractal-dimension analysis, benchmarking, multiple visualisations, or other fractals. The renderer should save `outputs/sierpinski_baseline.png`, select CUDA when available, use `depth = 7` and `resolution = 3 ** depth`, build the carpet with vectorised PyTorch tensor operations using ternary digit tests, validate against `8 ** depth`, render a black-and-white matplotlib image, and keep the code simple. The Slurm job should use the `comp3710` partition/account, request one GPU, use a five-minute time limit, avoid memory requests, write output/error logs under `outputs/`, activate the `torch` conda environment, set `MPLBACKEND=Agg`, and run `python src/sierpinski.py`.
- Assistance summary: Added a Stage 1 PyTorch baseline renderer and a Rangpur Slurm job script.
- Implementation approach: Used one-dimensional integer coordinate tensors, broadcast ternary digit masks into a two-dimensional Boolean removal mask at each recursion level, and rendered the final Boolean tensor as a black-retained, white-removed image.
- Issues encountered: Local execution could not be completed because this shell has no `python` command, its `python3` environment lacks `matplotlib`, and `$HOME/miniconda3` is not available here. Syntax checks for the Python and Slurm files passed.
- Changes made: Created `src/sierpinski.py`, created `jobs/render_job.sh`, and documented the prompt and implementation work in `prompts.md`.

## Depth Progression Visualisation

- Prompt: Implement the next stage of the COMP3710 Sierpiński carpet project: visualise how the fractal develops across recursion depths. Create `src/depth_progression.py` and `jobs/depth_progression_job.sh`; do not modify `src/sierpinski.py`; generate depths 1 through 6 using `resolution = 3 ** depth`; use the same vectorised PyTorch base-3 membership algorithm; verify each retained pixel count against `8 ** depth`; create a single 2 x 3 black-and-white figure; save `outputs/sierpinski_depth_progression.png`; keep the implementation simple; do not implement box-counting, benchmarks, or alternative fractals.
- Assistance summary: Added a depth progression renderer and a Rangpur Slurm job script for the progression figure.
- Implementation approach: Reused the tensor-based ternary digit test in a small `generate_carpet(depth, device)` function, then plotted depths 1 to 6 in a 2 x 3 matplotlib grid.
- Issues encountered: None during implementation.
- Changes made: Created `src/depth_progression.py`, created `jobs/depth_progression_job.sh`, and documented this stage in `prompts.md`.

## Box-Counting Fractal Dimension

- Prompt: Implement the box-counting fractal-dimension analysis stage of the UQ COMP3710 Sierpiński carpet project. Create `src/fractal_dimension.py` and `jobs/fractal_dimension_job.sh`; do not modify the existing baseline or depth-progression programs; generate the depth-7 carpet with the same vectorised PyTorch base-3 membership algorithm; count occupied boxes for box sizes `1, 3, 9, 27, 81, 243, 729` using PyTorch reshape/view/permute operations; fit `log(N)` against `log(1 / epsilon)` using `np.polyfit`; plot measured points and fitted line; save `outputs/sierpinski_fractal_dimension.png`; do not implement benchmarking, alternative visualisations, other fractals, classes, or extra abstractions.
- Assistance summary: Added a PyTorch box-counting analysis script and a Rangpur Slurm job script for estimating the Sierpiński carpet dimension.
- Implementation approach: Generated a depth-7 Boolean mask on the selected device, counted occupied boxes directly from the PyTorch mask using non-overlapping tensor blocks, converted only the final log values to NumPy for `np.polyfit`, and plotted the fitted relationship.
- Issues encountered: Full local plotting could not be run because this shell's `python3` environment lacks `matplotlib`; a PyTorch-only smoke test of generation, occupied-box counts, and the fitted slope passed.
- Changes made: Created `src/fractal_dimension.py`, created `jobs/fractal_dimension_job.sh`, and documented this stage in `prompts.md`.

## CPU Versus GPU Benchmark

- Prompt: Implement the CPU versus GPU performance benchmark stage of the UQ COMP3710 Sierpiński carpet project. Create `src/benchmark.py` and `jobs/benchmark_job.sh`; do not modify the existing fractal, depth progression, or fractal-dimension programs; benchmark depths 4 through 8; time only the vectorised PyTorch fractal-generation computation; use one warm-up and three timed repetitions per depth; synchronize CUDA timing; verify every result against `8 ** depth`; print CPU/GPU timing tables and speedup when CUDA is available; plot execution time versus resolution; save `outputs/sierpinski_cpu_gpu_benchmark.png`; do not add other fractals, unrelated optimisations, classes, or extra scope.
- Assistance summary: Added a CPU/GPU benchmark script and a Rangpur Slurm job script for timing the PyTorch Sierpiński carpet generation.
- Implementation approach: Reused the same vectorised tensor algorithm on CPU and GPU, timed coordinate creation, Boolean mask creation, and recursive tensor operations only, synchronized CUDA before and after timed GPU runs, and kept plotting outside the measured section.
- Issues encountered: Full local plotting could not be run because this shell's `python3` environment lacks `matplotlib`; a PyTorch-only CPU smoke test for depths 4 through 8 passed correctness checks.
- Changes made: Created `src/benchmark.py`, created `jobs/benchmark_job.sh`, and documented this benchmark stage in `prompts.md`.

## Final README

- Prompt: Finalize `README.md` for the UQ COMP3710 Sierpiński Carpet project, accurately describing the current implementation, mathematical construction, PyTorch base-3 algorithm, Rangpur GPU execution, generated results, self-similarity visualisation, box-counting dimension analysis, CPU/GPU benchmark results, project structure, running instructions, AI-assisted development, and key findings. Do not modify Python code, Slurm scripts, output images, or analysis results.
- Assistance summary: Replaced the placeholder README with a concise final project description and included the existing generated figures and measured benchmark table.
- Implementation approach: Documented only implemented project stages and used the user-provided benchmark results without inventing additional measurements.
- Issues encountered: None.
- Changes made: Updated `README.md` and documented this README-editing step in `prompts.md`.
