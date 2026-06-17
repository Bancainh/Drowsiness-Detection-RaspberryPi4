# The Personalized Edge-Based Driver Drowsiness Detection on Raspberry Pi 4 Using Dynamic EAR Calibration

## Overview
[cite_start]This repository contains the official implementation and data processing pipeline for a real-time driver fatigue monitoring system deployed on an embedded edge platform (Raspberry Pi 4)[cite: 1, 8]. [cite_start]The system utilizes the lightweight MediaPipe Face Mesh framework to extract 3D facial landmarks and compute the Eye Aspect Ratio (EAR)[cite: 9, 67, 74]. 

[cite_start]To handle inter-subject variability and overcome the limitations of rigid, fixed thresholds, we integrate a personalized, adaptive threshold calibration mechanism[cite: 10, 26]. [cite_start]Operating entirely offline, the system achieves a stable processing speed of approximately **18.9 FPS** and an overall accuracy of **86.95%**[cite: 12].



## Core Technical Features
* [cite_start]**Personalized Calibration:** Automatically collects EAR data series during the initial 15.0 seconds of operation, filtering out noise ($\le 0.05$) to establish a customized baseline[cite: 86, 89, 90]. [cite_start]The adaptive threshold is computed mathematically as: $Th_{adapt} = \mu_{baseline} \times 0.8$[cite: 92].
* [cite_start]**Temporal Decision Logic:** Triggers a drowsiness alert only when the EAR drops below the adaptive threshold for a continuous duration of 1.2 seconds (equivalent to ~23 consecutive frames at 18.9 FPS) to avoid false triggers from normal blinking[cite: 94, 95, 96].
* [cite_start]**Robust Forward-Fill Logic:** Minimizes false negatives during critical movements (such as head-nodding or extreme shaking) by automatically retaining the last valid EAR state for up to 5 consecutive frames during tracking loss[cite: 69, 70, 99].

## Repository Structure
* `src/`: Core Python source code for data preparation, execution, tuning, and evaluation.
* [cite_start]`docs/`: Evaluation graphics, confusion matrices, and the published research paper[cite: 111, 145, 175].

## Dataset Specification
[cite_start]The model was validated on a dataset consisting of **124,988 frames** and **77 video sequences** recorded across four distinct environmental conditions[cite: 106]:
1. [cite_start]**Day_Light:** 26 cases (46,442 frames) [cite: 107]
2. [cite_start]**Day_Light_Glasses:** 15 cases (21,837 frames) [cite: 108]
3. [cite_start]**Night_Light:** 19 cases (32,123 frames) [cite: 108, 109]
4. [cite_start]**Night_Light_Glasses:** 17 cases (24,586 frames) [cite: 109]

Due to file size constraints on GitHub, the complete raw dataset along with the validation files can be fully downloaded from our managed storage:
👉 **[Download Dataset & Verification Files via Google Drive]** *https://drive.google.com/drive/folders/1SkD5U9TYQa9wGh1YvhCKU5nSEBzf5ol_*

## Pipeline Execution Order
To reproduce the experimental results documented in our paper, please execute the scripts in the `src/` directory sequentially:

1. [cite_start]**`darkness_simulation.py`**: Applies non-linear Gamma transformation to simulate low-light environments from daylight recordings[cite: 105].
2. **`main_raspberry_pi.py` (Mode 1)**: Batches processing on raw videos to extract mathematical vectors and output raw machine predictions (`_Pred.csv`).
3. **`ground_truth_annotator.py`**: A custom analytical tool utilized for human gán nhãn (ground truth definition) that automatically isolates the initialization/calibration window.
4. **`merge_folder_to_csv.py` & `build_ultimate_dataset.py`**: Matches, cleanses, and aligns machine predictions and human labels to construct the `ULTIMATE_MASTER_DATASET.csv`.
5. **`count_unique_users.py` & `count_users_per_condition.py`**: Validates the dataset structural integrity and extracts overall metadata characteristics.
6. **`grid_search_tuning.py`**: Automatically iterates through the parameter search space to prove that the 1.2s delay provides the optimal F1/Recall trade-off.
7. [cite_start]**`evaluate_metrics.py` & `evaluate_all_conditions.py`**: Computes standard performance metrics (Accuracy, Precision, Recall) and outputs the corresponding confusion matrices across all 4 scenarios[cite: 175].
8. [cite_start]**`plot_fps_chart.py` & `plot_benchmark_comparison.py`**: Draws execution speed profiles and maps accuracy benchmarks against related embedded works[cite: 111, 179].
9. [cite_start]**`main_raspberry_pi.py` (Mode 2)**: Runs the live inference loop using connected video capture nodes for live vehicle monitoring[cite: 52].

## Performance Benchmarking
[cite_start]As shown in our paper, moving from a fixed Dlib architecture (0.23 threshold) to our lightweight, adaptive MediaPipe framework on a Raspberry Pi 4 significantly drops false-alarm rates for diverse facial patterns while preserving hardware efficiency[cite: 38, 39, 43, 84, 190].

| Study (Ref) | Hardware | Method | Accuracy (%) | FPS | Critical Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Thanh [9] | RPi 3 | SVM | >86% | Low | [cite_start]Unstable under low-light profiles [cite: 180] |
| Vadlamudi [10] | RPi 4 | CNN+LSTM | ~93% | 0.86 | [cite_start]High Latency; not real-time [cite: 180] |
| Ming [12] | Xavier | Adaptive | >95% | 34 | [cite_start]High hardware acquisition costs [cite: 180] |
| Sharma [11] | RPi 4 | Fixed (0.23) | 85% | 30 | [cite_start]Low flexibility for specific eye shapes [cite: 180] |
| **Ours** | **RPi 4** | **Adaptive** | **86.95%** | **18.9** | [cite_start]**Best performance/cost trade-off on edge** [cite: 180] |

## Citation
If you utilize this work, codebase, or dataset in your research, please cite our paper:
```text
Nguyen Ba Chinh, Pham Van Hoang Anh Tu, Le Van Sang, Dang Van Hieu. "The Personalized Edge-Based Driver Drowsiness Detection on Raspberry Pi 4 Using Dynamic EAR Calibration." FPT University, 2026.
