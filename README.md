# 🧪 DexBioLab

[![Paper](https://img.shields.io/badge/arXiv-Paper-F5F5F7?style=flat-square&logo=arxiv&logoColor=111111&labelColor=F5F5F7&color=E8E8ED)](https://arxiv.org/abs/2610.13643)
[![Project Page](https://img.shields.io/badge/Project-Page-F5F5F7?style=flat-square&logo=safari&logoColor=111111&labelColor=F5F5F7&color=E8E8ED)](https://cumulusalpha.github.io/DexBioLabWebPage/)

---

**Benchmarking Dexterous Manipulation with Dynamic Experimental Phenomena in Realistic Laboratories**

![DexBioLab Teaser](assets/teaser.gif)

Simulate laboratory manipulation tasks, collect scripted demonstrations, render
LeRobot datasets, and evaluate learned policies.

The commands below use **Linux / Bash with an NVIDIA GPU**. Run them from the
DexBioLab repository root unless a command explicitly changes directories.

## 📥 1. Download the repository

```bash
git clone https://github.com/CumulusAlpha/DexBioLab.git
cd DexBioLab
```

## 🛠️ 2. Install the environment

With the NVIDIA CUDA apt repository configured, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y cuda-compiler-12-8
uv sync
bash assets/scenes/download_scenes.sh
```

Optional — video rendering also requires `ffmpeg`:

```bash
sudo apt-get install -y ffmpeg
```

## 🎮 3. Verify installation

```bash
uv run python scripts/list_envs.py
uv run python scripts/play.py XHand-Tube-Insert
```

Open **http://localhost:8080** in your browser. If the scene loads and the
scripted robot performs the tube-insertion task, the simulator is ready ✅.
The Viser **Task Goal** panel shows the current goal and its language prompt.
Stop the process with `Ctrl+C` before continuing.

The examples below use `Shadow-Tube-Insert`; replace it with another
registered task ID when needed.

## 📦 4. Collect demonstrations

Run the scripted policy collector:

```bash
uv run python scripts/collect.py Shadow-Tube-Insert \
  --num-envs 256 \
  --num-episodes 100
```

The output is a state-only replay buffer:

```text
logs/collect/Shadow-Tube-Insert/replay_buffer.zarr
```

## 🎥 5. Render LeRobot datasets

Replay the collected simulation states and stream camera frames directly into
a LeRobot v3.0 dataset:

```bash
uv run python scripts/render_lerobot.py \
  --input-zarr logs/collect/Shadow-Tube-Insert/replay_buffer.zarr \
  --format openpi \
  --num-envs 256 \
  --width 224 \
  --height 224
```

📁 The final dataset defaults to:

```text
thirdparty/openpi/datasets/dexbiolab_lerobot/Shadow-Tube-Insert
```

🔍 Inspect the rendered dataset interactively:

```bash
uv run python scripts/data_vis.py Shadow-Tube-Insert \
  --input thirdparty/openpi/datasets/dexbiolab_lerobot/Shadow-Tube-Insert
```

Open the Viser URL printed in the terminal and select an episode. You can also
pass the collection's `replay_buffer.zarr` to `--input` to inspect it before
rendering.

## 🧠 6. Evaluate an OpenPI policy

Evaluation requires a **trained checkpoint for the same task**. OpenPI training
settings are in [`thirdparty/openpi/config.yaml`](thirdparty/openpi/config.yaml);
its source and further documentation are in
[`thirdparty/openpi`](thirdparty/openpi/README.md).

Install the standalone OpenPI environment for policy serving:

```bash
cd thirdparty/openpi
bash install.bash
cd ../..
```

### 🚀 Terminal A: start the policy server

Replace `<experiment>` and `<step>` with your actual checkpoint directories.
The configuration name is the task ID:

```bash
cd thirdparty/openpi
uv run python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config Shadow-Tube-Insert \
  --policy.dir "checkpoints/pi05_dexbiolab_ckpts/Shadow-Tube-Insert/<experiment>/<step>"
```

Keep this server running while the evaluator connects to it.

### 🤖 Terminal B: run evaluation and save replay states

From the DexBioLab root:

```bash
mkdir -p logs/openpi_eval/Shadow-Tube-Insert
set -o pipefail
uv run python scripts/evaluate_openpi.py \
  --task-id Shadow-Tube-Insert \
  --host 127.0.0.1 --port 8000 \
  --num-envs 100 --num-episodes 100 \
  --seed 0 \
  --viewer none \
  --camera-width 224 --camera-height 224 \
  --diagnostics-path logs/openpi_eval/Shadow-Tube-Insert/episodes.jsonl \
  --replay-output-dir logs/openpi_eval_replays/Shadow-Tube-Insert \
  --replays-per-outcome 8 \
  2>&1 | tee logs/openpi_eval/Shadow-Tube-Insert/evaluation.log
```

This matches the Gaussian, 224 × 224 rendering example above. Match
camera settings to your training data. Gaussian rendering requires the
corresponding scene assets.

Evaluation runs **one initial episode per environment**, freezes completed
environments, and retires them from policy inference. Headless evaluation
requires `--num-envs >= --num-episodes`; to run a smaller evaluation, reduce
both values together.

💾 Replay recording requires `--viewer none`. It saves compressed `.npz`
simulation states including initial and terminal frames, with up to eight
successes and eight failures. If an outcome has too few episodes, the other
outcome fills the remaining slots when available. Use a separate output
directory for each evaluation run to keep results distinct.

For live policy inspection, use `--viewer viser` and omit
`--replay-output-dir`. Use the headless command above to obtain success statistics.

## 📊 7. Check the success rate

The evaluator prints a final summary, for example:

```text
Evaluation complete: episodes=100, successes=75, success_rate=75.0%, steps=...
```

Retrieve it from the saved log:

```bash
rg "Evaluation complete:" logs/openpi_eval/Shadow-Tube-Insert/evaluation.log
```

**Success rate = successful episodes / completed episodes.** Check that
`episodes` equals the requested count; `--max-steps` can stop evaluation early.
The log also reports termination reasons and success-condition diagnostics.
`episodes.jsonl` records each completed episode's success, length, prompt,
termination reasons, and policy diagnostics.

The replay directory's `manifest.tsv` describes the **selected replay clips**.
It is balanced by outcome and should not be used to calculate the overall
evaluation success rate.

## 🔁 8. Render replay states and inspect policy rollouts

After evaluation, render the saved states into MP4 videos:

```bash
uv run python scripts/render_episode_replays.py \
  logs/openpi_eval_replays/Shadow-Tube-Insert \
  logs/openpi_eval_videos/Shadow-Tube-Insert \
  --width 640 --height 480
```

🎬 Open the generated `.mp4` files in
`logs/openpi_eval_videos/Shadow-Tube-Insert/`. Each video shows an episode's
configured camera views side by side. Filenames include the actual `success`
or `failure` outcome, episode number, and environment ID; `manifest.tsv`
provides the mapping.

The renderer restores recorded scene states and task effects, including mixer
displays, Bunsen flames, and heating effects. It uses standard cameras and
does not require the policy server or checkpoint. These videos visualize the
recorded trajectories; their pixels may differ from Gaussian evaluation images.
Add `--overwrite` to regenerate an already rendered directory.

