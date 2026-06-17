# The Personalized Edge-Based Driver Drowsiness Detection on Raspberry Pi 4 Using Dynamic EAR Calibration

## Overview
This repository contains the official implementation and data processing pipeline for a real-time driver fatigue monitoring system deployed on an embedded edge platform (Raspberry Pi 4). The system utilizes the lightweight MediaPipe Face Mesh framework to extract 3D facial landmarks and compute the Eye Aspect Ratio (EAR). 

To handle inter-subject variability and overcome the limitations of rigid, fixed thresholds, we integrate a personalized, adaptive threshold calibration mechanism. Operating entirely offline, the system achieves a stable processing speed of approximately **18.9 FPS** and an overall accuracy of **86.95%**.

## Core Technical Features
* **Personalized Calibration:** Automatically collects EAR data series during the initial 15.0 seconds of operation, filtering out noise ($\le 0.05$) to establish a customized baseline. The adaptive threshold is computed mathematically as: $Th_{adapt} = \mu_{baseline} \times 0.8$.
* **Temporal Decision Logic:** Triggers a drowsiness alert only when the EAR drops below the adaptive threshold for a continuous duration of 1.2 seconds (equivalent to ~23 consecutive frames at 18.9 FPS) to avoid false triggers from normal blinking.
* **Robust Forward-Fill Logic:** Minimizes false negatives during critical movements (such as head-nodding or extreme shaking) by automatically retaining the last valid EAR state for up to 5 consecutive frames during tracking loss.

## Repository Structure
* `src/`: Core Python source code for data preparation, execution, tuning, and evaluation.
* `docs/`: Evaluation graphics, confusion matrices, and the published research paper.

## Dataset Specification
The model was validated on a dataset consisting of **124,988 frames** and **77 video sequences** recorded across four distinct environmental conditions:
1. **Day_Light:** 26 cases (46,442 frames)
2. **Day_Light_Glasses:** 15 cases (21,837 frames)
3. **Night_Light:** 19 cases (32,123 frames)
4. **Night_Light_Glasses:** 17 cases (24,586 frames)

Due to file size constraints on GitHub, the complete raw dataset along with the validation files can be fully downloaded from our managed storage:
👉 **[Download Dataset & Verification Files via Google Drive](https://drive.google.com/drive/folders/1SkD5U9TYQa9wGh1YvhCKU5nSEBzf5ol_)**

## Pipeline Execution Order
To reproduce the experimental results documented in our paper, please execute the scripts in the `src/` directory sequentially:

1. **`darkness_simulation.py`**: Applies non-linear Gamma transformation to simulate low-light environments from daylight recordings.
2. **`main_raspberry_pi.py` (Mode 1)**: Batches processing on raw videos to extract mathematical vectors and output raw machine predictions (`_Pred.csv`).
3. **`ground_truth_annotator.py`**: A custom analytical tool utilized for human labeling (ground truth definition) that automatically isolates the initialization/calibration window.
4. **`merge_folder_to_csv.py` & `build_ultimate_dataset.py`**: Matches, cleanses, and aligns machine predictions and human labels to construct the `ULTIMATE_MASTER_DATASET.csv`.
5. **`count_unique_users.py` & `count_users_per_condition.py`**: Validates the dataset structural integrity and extracts overall metadata characteristics.
6. **`grid_search_tuning.py`**: Automatically iterates through the parameter search space to prove that the 1.2s delay provides the optimal F1/Recall trade-off.
7. **`evaluate_metrics.py` & `evaluate_all_conditions.py`**: Computes standard performance metrics (Accuracy, Precision, Recall) and outputs the corresponding confusion matrices across all 4 scenarios.
8. **`plot_fps_chart.py` & `plot_benchmark_comparison.py`**: Draws execution speed profiles and maps accuracy benchmarks against related embedded works.
9. **`main_raspberry_pi.py` (Mode 2)**: Runs the live inference loop using connected video capture nodes for live vehicle monitoring.

## Performance Benchmarking
As shown in our paper, moving from a fixed Dlib architecture (0.23 threshold) to our lightweight, adaptive MediaPipe framework on a Raspberry Pi 4 significantly drops false-alarm rates for diverse facial patterns while preserving hardware efficiency.

| Study (Ref) | Hardware | Method | Accuracy (%) | FPS | Critical Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Thanh [9] | RPi 3 | SVM | >86% | Low | Unstable under low-light profiles |
| Vadlamudi [10] | RPi 4 | CNN+LSTM | ~93% | 0.86 | High Latency; not real-time |
| Ming [12] | Xavier | Adaptive | >95% | 34 | High hardware acquisition costs |
| Sharma [11] | RPi 4 | Fixed (0.23) | 85% | 30 | Low flexibility for specific eye shapes |
| **Ours** | **RPi 4** | **Adaptive** | **86.95%** | **18.9** | **Best performance/cost trade-off on edge** |

## Citation
If you utilize this work, codebase, or dataset in your research, please cite our paper:
```text
Nguyen Ba Chinh, Pham Van Hoang Anh Tu, Le Van Sang, Dang Van Hieu. "The Personalized Edge-Based Driver Drowsiness Detection on Raspberry Pi 4 Using Dynamic EAR Calibration." FPT University, 2026.
