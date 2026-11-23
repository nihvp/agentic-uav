# Reproducing "Agentic UAVs" (Koubaa & Gabr, arXiv 2509.13352v2): results

Branch `paper-sim`. Method and implementation details: [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md).
Example frames with every system's output: [figures/paper/examples.md](figures/paper/examples.md).
Full scorer output with all tests: [figures/paper/summary.md](figures/paper/summary.md).

## What the paper claims

A UAV with an LLM reasoning layer ("agentic") is compared with a rule-based YOLO
detector on simulated search-and-rescue scenes: a crowd, and in emergencies one person
lying motionless away from it. On 44 samples per system, the agentic systems were
slower but detected more people, recommended actions and described the context, which
YOLO alone cannot do.

## What we ran

| | Paper | Here |
|---|---|---|
| Simulator | Gazebo Harmonic, PX4 1.14.3, ROS 2 Humble | Gazebo Harmonic, PX4 1.16.0, ROS 2 Jazzy |
| Drone and camera | RealSense D455, 30 Hz | x500 with RGB-D camera, 640x360, pitched 45 deg down, 6 m up |
| Scenes | Hajj crowd (white clothing) | 44 generated scenes (fixed seed): 6-12 standing people; in 22 emergency scenes one more person lying at least 2 m from the crowd |
| Rule-based system | YOLOv11 | YOLOv11s, class person |
| Agentic systems | GPT-4 (cloud), Gemma-3 4B (local) | `qwen3-vl:8b` (local, Ollama, vision + thinking): image + YOLO detections in, structured JSON out |
| Hardware | not stated | one RTX 5070 Ti 16 GB shared by Gazebo, YOLO and the LLM |

Pipeline, all automated:

1. **Scenes**: generated from a seed; the scene file is the ground truth, read only by
   the spawner and the scorer.
2. **Spawn**: the scene's people are placed in Gazebo; the spawner checks that exactly
   that scene's people are present.
3. **Capture**: the drone flies there through the normal flight-control layer (takeoff,
   navigate, steady hover) and saves the camera frames. It hovered 0.002-0.31 m from the
   planned spot, heading within 0.33 deg.
4. **Analyse**: every system gets the same saved frame. No system sees the ground truth.
5. **Score**: the paper's metrics and tests, plus correctness against the ground truth,
   which the paper does not report.

## Results

44 samples per system. Paper numbers in brackets (rule-based / Gemma-3 local / GPT-4).

| Metric | Rule-based (YOLO) | Agentic (`qwen3-vl:8b`) | Test |
|---|---|---|---|
| Processing time, mean (SD) | 0.018 s (0.006) [29.5 us / 1.48 s / 4.95 s] | 21.6 s (10.1) [1.48 s local] | Mann-Whitney p < 0.001 |
| Slower than YOLO | 1x | ~1,200x [~50,000x / 168,000x] | |
| Detection confidence, mean | 0.899 [0.716] | 0.940 [0.760 / 0.790] | Mann-Whitney p < 0.001 |
| Persons detected (of 44 scenes) | 44 (100%) [33 / 37 / 40] | 44 (100%) | identical |
| Action recommendation rate (ARR) | 0% [0%] | 100% [79% / 92%] | chi-square p < 0.001 |
| Contextual analysis rate (CAR) | 0% [0%] | 100% [88% / 94%] | chi-square p < 0.001 |

Correctness against the ground truth (not in the paper):

| Metric | Rule-based (YOLO) | Agentic (`qwen3-vl:8b`) |
|---|---|---|
| Situation (normal / emergency) correct | n/a: no assessment | 44/44 (100%) |
| Emergencies caught | n/a | 22/22 (100%) |
| False alarms in normal scenes | n/a | 0/22 |
| Correct response (rescue/alert only in emergencies) | 22/44 (50%) | 44/44 (100%) |
| Person count, mean absolute error | 0.36 | 0.20 |
| Unparseable replies | n/a | 0 |

Repeatability: a second run of the same model on the same frames gave the same verdict
on 43 of 44 scenes. On the 44th (scene 18) the reply was not valid JSON after 159 s,
which the scorer counts as a miss.

## What this shows

* **The paper's main claim reproduces with a local model**: the agentic system adds
  what YOLO cannot, namely a situation assessment, recommended actions and a scene
  description, at a large cost in time. All differences are significant at p < 0.001.
* **It was also correct**, which the paper does not measure: every emergency caught, no
  false alarms, and a better person count than YOLO. Example, scene 1
  ([frame and replies](figures/paper/examples.md)): YOLO found 12 of the 13 people; the
  model counted 13 and reported "a person lying motionless on the ground to the left of
  the crowd", recommending ALERT_EMERGENCY_SERVICES.
* **Speed is the cost**: 22 s per sample here, against the paper's 1.5 s for its local
  model. The diagnostics explain it: `qwen3-vl:8b` writes about 2,100 tokens of
  reasoning before its answer, and the length varies (6-159 s per sample).

## Limitations

* **Easy scenes**: one clearly isolated person lying on bare ground. Perfect accuracy
  partly reflects that; YOLO also found people in every frame, unlike the paper's 75%.
* **One agentic system**: no cloud GPT-4 tier (local only, by choice). The paper's own
  local model (`gemma3:4b`) and a non-thinking build are not run yet.
* **Different people**: generic Gazebo person models, not a Hajj crowd.
* **Independent-samples statistics**, as in the paper, although all systems saw the
  same frames; paired tests would be stricter.
* **ARR counts any action**, including "continue monitoring" in normal scenes, so it
  measures willingness to recommend, not correctness. The correct-response row does
  that.

## Notes on the paper

* YOLO at 29.5 us per sample is too fast for YOLOv11 inference (ours: 18 ms on a
  GPU, including preprocessing); it probably times only post-processing. That is why the
  paper's slowdown factors (~50,000x, 168,000x) are so large.
* The text says agentic UAVs are "105 times slower", while Table 2 says ~50,000x and
  168,000x.
* LLM confidence is self-reported, not calibrated, so comparing it with YOLO's score
  is not like with like.

## Next steps

1. Harder scenes: a lying person close to or partly hidden by the crowd, sitting
   people, other altitudes, so that correctness separates the systems.
2. More systems: the paper's own `gemma3:4b`, and a non-thinking `qwen3-vl` build for
   the speed/accuracy trade-off.
3. The response step end to end: land-and-deploy and an alert message for detected
   emergencies.
