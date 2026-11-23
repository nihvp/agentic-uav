# Reproducing "Agentic UAVs" (Koubaa & Gabr, arXiv 2509.13352v2)

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
   (the ground truth). Every scene is laid out around the point the camera looks at, so
   all its people are in view.
2. **Spawn** (`agentic_uav/experiment/spawn.py`, runs in the `sim` container): clears the
   previous scene's people and spawns the next scene's through Gazebo's create/remove
   services. Scene setup only; nothing moves during a sample, and no agent layer reads
   the scene file.
3. **Capture** (next): fly to the scene's viewpoint, save a burst of frames + depth + PX4
   pose per scene to `experiments/paper/captures/`.
4. **Analyse** (next), offline on the saved frames so every system sees identical input:
   * rule-based: YOLO only (detections, no actions), as in the paper;
   * agentic: YOLO detections + the image to the vision model, structured JSON output
     (`person_detected`, `confidence`, `situation`, `surrounding_features`,
     `recommended_actions`, ...), timed per sample.
5. **Respond** (next): for emergencies, the MCP alert tool and a land-and-deploy command;
   shown end to end in a separate demo run (landing in every sample would dominate time).
6. **Score** (next): the paper's metrics and tests, plus correctness from ground truth,
   which the paper does not report: emergency recall and false-alarm rate, correct
   "no intervention" in normal scenes, and person-count error.

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
4. Fly to that scene's viewpoint (from `experiments/paper/scenes.json`) and check that
   perception (`detector: color+yolo`) tracks `person` nodes.

## Caveats in the paper to note in the write-up

* YOLO at 29.5 us per sample is too fast for detection itself; probably post-processing.
* "105 times slower" in the text vs "~50k x" / "168k x" in Table 2.
* ARR counts any recommended action, also in normal scenes where none is correct;
  LLM "confidence" is self-reported, not calibrated.

## Status

* [x] Branch, plan
* [x] YOLOv11 detector option in perception (`detector: yolo` / `color+yolo`)
* [x] Scene generator + spawner
* [x] Vision check (`llm_check --vision`); model switch pending the VM check
* [ ] Capture, analysers, response, scoring
