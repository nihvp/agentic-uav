# Reproducing "Agentic UAVs" (Koubaa & Gabr, arXiv 2509.13352v2)

Results for presenting: [PAPER_RESULTS.md](PAPER_RESULTS.md). Live demo of one scene:
`scripts/demo_scene.sh [scene]`.

Branch `paper-sim`. Goal: run the paper's Section 4 experiment in this stack and report
our own results with the same metrics. The paper's numbers are the reference, not a
target to hit (some are inconsistent, see the end).

## The paper's experiment

* Gazebo Harmonic + PX4 SITL 1.14.3 + ROS 2 Humble; YOLOv11 on a simulated RealSense
  D455 at 30 Hz; GPT-4 API and a local Gemma-3 4B as reasoning layers.
* Scenes: a crowd (Hajj). *Normal*: upright people, correct response = no intervention.
  *Emergency*: one person lying motionless, isolated from the crowd; response =
  `LAND_AND_DEPLOY_RESCUE_KIT` plus an email alert with coordinates and an annotated image.
* n = 44 balanced samples per system, identical inputs: rule-based (YOLO only), agentic
  local, agentic cloud.
* Metrics: processing time per sample; detection confidence (YOLO's score, or the LLM's
  self-reported score); persons detected (of 44); Action Recommendation Rate (ARR,
  non-empty recommended actions); Contextual Analysis Rate (CAR, `surrounding_features`
  longer than 50 characters). Statistics: one-way ANOVA + Tukey HSD, Mann-Whitney U
  where normality fails, chi-square for rates, Cohen's d / rank-biserial, Shapiro-Wilk,
  Levene, alpha = 0.05.

## Our setup and differences

| Paper | Here | Note |
|-------|------|------|
| ROS 2 Humble, PX4 1.14.3 | ROS 2 Jazzy, PX4 1.16.0 | newer; report it |
| RealSense D455, 30 Hz | x500_depth OakD-Lite RGB-D, 640x360 @ 10 Hz | may raise resolution for YOLO at range |
| YOLOv11 | YOLOv11 (`ultralytics`), class `person` | same |
| GPT-4 (cloud) + Gemma-3 4B (local) | one local vision model, `qwen3-vl:8b` (Ollama) | user decision: local only. Optional extra local tier: `gemma3:4b`, the paper's own local model |
| Hajj crowd, white ihram | Gazebo Fuel person models, standing and lying | appearance differs; report it |
| Email alert | MCP alert tool to a local mail sink | real message, nowhere external |

Why the model changes: `qwen3:14b` reads text only, and the paper's agentic systems
analyse the camera image (that is how they detect persons YOLO missed). The 16 GB GPU
cannot hold `qwen3:14b` and a vision model next to Gazebo and YOLO, so one vision model
with tool calling serves both the image analysis and the flight tools. `qwen3-vl:8b`
(vision + tools + thinking, ~6 GB at 4-bit) is the candidate; it must pass
`llm_check` and `llm_check --vision` on the VM before it is adopted. Fallbacks:
`qwen2.5vl:7b`, `gemma3:12b` (vision, no tool calling).

## Pipeline

1. **Scenes** (`agentic_uav/experiment/scenes.py`): 44 scenes from a fixed seed, 22
   normal and 22 emergency in shuffled order, written to `experiments/paper/scenes.json`
   (the ground truth). Every scene is laid out around the point the camera looks at, and
   each person is placed only if the whole body (feet to head, or end to end when lying)
   projects inside the 640x360 frame with a 20 px margin. (The first layout checked only
   body centres; the first capture cut a lying person's head off at the frame edge.)
2. **Spawn** (`agentic_uav/experiment/spawn.py`, runs in the `sim` container): clears the
   previous scene's people and spawns the next scene's through Gazebo's create/remove
   services. Scene setup only; nothing moves during a sample, and no agent layer reads
   the scene file.
3. **Capture** (`agentic_uav/experiment/capture.py` in the agent container, batch:
   `scripts/capture_scenes.sh`): take off if needed, fly to the scene's viewpoint through
   the action layer, wait until the hover is steady (speed < 0.15 m/s and heading still
   for 2 s), save 3 frames 0.5 s apart to `experiments/paper/captures/scene_NN/`:
   `rgb_K.png`, `depth_K.npz` and `meta.json` (PX4 pose, camera pose and intrinsics per
   frame, offset from the planned viewpoint). It reads only the viewpoint from the scene
   file, and writes no label. `eval_true_pose.txt` (Gazebo's drone pose) is for scoring
   only.
4. **Analyse** (`agentic_uav/experiment/analyse.py`, agent container), offline on each
   scene's `rgb_0.png`, so every system sees identical input; reads no ground truth:
   * `--system rule`: YOLOv11 person detections only (count, max confidence), no
     assessment or actions, as the paper's rule-based baseline;
   * `--system agentic`: the YOLO detections plus the image to the vision LLM
     (`--model`, default `LLM_MODEL`), temperature 0, JSON mode; the prompt states the
     mission (find people in distress, e.g. collapsed or lying motionless) and asks for
     `person_detected`, `persons_count`, `confidence`, `situation`,
     `emergency_description`, `surrounding_features`, `recommended_actions` (from
     NO_INTERVENTION, CONTINUE_MONITORING, HOVER_AND_OBSERVE, LAND_AND_DEPLOY_RESCUE_KIT,
     ALERT_EMERGENCY_SERVICES).
   One untimed warm-up on a blank frame first; per sample it records YOLO time, LLM time,
   total processing time, raw reply, tokens. Output: `results/rule.jsonl`,
   `results/agentic_<model>.jsonl` (resumable). An optional extra tier is the paper's own
   local model: `--model gemma3:4b` (pull it first).
5. **Respond** (next): for emergencies, the MCP alert tool and a land-and-deploy command;
   shown end to end in a separate demo run (landing in every sample would dominate time).
6. **Score** (`agentic_uav/experiment/score.py`, agent container; scipy comes with
   ultralytics): joins `results/*.jsonl` with `scenes.json` and writes
   `results/summary.md` / `summary.json`.
   * Paper metrics: processing time, detection confidence, persons detected, ARR, CAR;
     Shapiro-Wilk, Levene, ANOVA + Tukey HSD (3+ systems) or Welch's t (2), Mann-Whitney
     U, Cohen's d and rank-biserial r (same sign), chi-square / Fisher for rates. Like
     the paper, systems are treated as independent samples although they saw the same
     scenes.
   * Correctness (not in the paper): situation accuracy, emergency recall, false-alarm
     rate, correct response (emergency: LAND_AND_DEPLOY_RESCUE_KIT or
     ALERT_EMERGENCY_SERVICES; normal: neither; so the rule baseline, which recommends
     nothing, scores 50 %), person-count error, unparseable replies.

## Run on the VM (current steps)

1. Use the vision model: in `.env` set `LLM_MODEL=qwen3-vl:8b`, then restart the stack
   (`scripts/run.sh`; the `llm` service pulls it, ~6 GB).
2. Model check, inside the agent container (not on the VM host, which has no
   `agentic_uav` package); both lines must print PASS:
   `docker compose exec agent bash -c "source /ws/install/setup.bash && python3 -m agentic_uav.reasoning.llm_check --vision"`
3. Spawn a scene and look at it in the Gazebo window (first spawn downloads the person
   model from Fuel): `docker compose exec sim python3 -m agentic_uav.experiment.spawn --scene 1`
   (on the VM). Gazebo's create service replies before the model exists, so the spawner
   waits up to `--wait` seconds (180) for all of them to appear. Check the lying person
   lies flat on the ground, not half buried or floating.
4. Capture one scene, then look at its frames (on the VM host):
   `scripts/capture_scenes.sh 1`, then open `experiments/paper/captures/scene_01/rgb_0.png`.
   All 13 people should be in the frame. Then all scenes: `scripts/capture_scenes.sh`
   (skips scenes already captured; re-run to retry failures).

## First results (run 1, 2026-11-23)

Rule (YOLOv11s) vs agentic (`qwen3-vl:8b`), 44 scenes each:

| metric | rule | agentic |
|---|---|---|
| processing time, mean (SD) s | 0.018 (0.006) | 21.6 (10.1) |
| detection confidence, mean | 0.899 (YOLO) | 0.940 (self-reported) |
| persons detected | 44/44 | 44/44 |
| ARR / CAR | 0 % / 0 % | 100 % / 100 % |
| situation accuracy, emergency recall, false alarms | n/a | 44/44, 22/22, 0/22 |
| person count, mean abs error | 0.36 | 0.20 |

All differences in time, confidence, ARR, CAR and correct response significant
(p < 0.001). Notes:
* Scene 32 was first captured with scene 31's 9 people still present (the spawner's
  model list timed out and read as an empty world). The spawner now refuses to continue
  unless the world holds exactly the scene's people; scene 32 was recaptured and
  re-analysed (`--only 32`), and the table above includes the corrected sample.
* The agentic time (22 s) is far above the paper's local model (seconds) because
  qwen3-vl thinks before answering: 2139 completion tokens and 8290 characters of
  thinking per sample on average.
* `--no-think` did not turn it off: the "nothink" run thought just as much (8817 chars,
  23.2 s). `qwen3-vl:8b` behaves as the Thinking build of Qwen3-VL, which has no off
  switch; analyse.py now stops a `--no-think` run when the model still thinks. That run
  is therefore a second run of the thinking setup (repeatability): it matched run 1
  on every scene except scene 18, where the reply was unparseable after 159 s (scored
  as a miss). For a non-thinking comparison, use an Instruct build
  (`qwen3-vl:8b-instruct`, if Ollama has that tag) as its own system.
* Ceiling effect: the agentic system is perfect on these scenes (one clearly isolated
  lying person on bare ground). Harder variants (lying person near or partly hidden by
  the crowd, sitting people, other altitudes) would make correctness discriminate.

## Caveats in the paper to note in the write-up

* YOLO at 29.5 us per sample is too fast for detection itself; probably post-processing.
* "105 times slower" in the text vs "~50k x" / "168k x" in Table 2.
* ARR counts any recommended action, also in normal scenes where none is correct;
  LLM "confidence" is self-reported, not calibrated.

## Status

* [x] Branch, plan
* [x] YOLOv11 detector option in perception (`detector: yolo` / `color+yolo`)
* [x] Scene generator + spawner
* [x] Vision check (`llm_check --vision`): `qwen3-vl:8b` passed tools and vision on the
  VM (warm image answer 0.6-1.3 s; the first one 27 s while loading, so analysers send an
  untimed warm-up request first)
* [x] Scene 1 spawned on the VM: 13/13 people, lying person flat on the ground
* [x] Capture node + batch script: scene 1 captured on the VM (0.2 m from the planned
  viewpoint, steady after 7 s, all people in frame)
* [x] Analysers (untested on the VM)
* [x] Scoring (tested on synthetic results)
* [ ] Response demo
