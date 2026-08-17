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
