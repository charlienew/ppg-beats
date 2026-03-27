# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPG-beats is a MATLAB research toolbox for detecting heartbeats in photoplethysmogram (PPG) signals. It provides ~20+ beat detection algorithms and a benchmarking framework to compare their performance across multiple public physiological datasets.

## Development Notes

There is no MATLAB build system or automated test suite — this is a research toolbox validated manually. Documentation is hosted on ReadTheDocs and auto-generated from MATLAB function comments using the `docs/functions/` Markdown files.

## Architecture

### Core Entry Points

- `source/detect_ppg_beats.m` — Main PPG beat detection wrapper; routes to a named algorithm via `eval()`, then calls `tidy_beats.m` for post-processing
- `source/detect_ecg_beats.m` — ECG beat detection wrapper using dual detectors (jqrs, rpeakdetect, gqrs) for quality validation by agreement
- `source/assess_beat_detectors.m` — Full benchmarking pipeline: detects beats, assesses ECG reference, time-aligns signals, computes performance metrics, and generates tables/figures
- `source/assess_multiple_datasets.m` — Runs `assess_beat_detectors` across multiple datasets

### Algorithm Plugin System

All PPG beat detectors live in `source/` as `*_beat_detector.m` files. Each has a standardized signature:
```matlab
function [peaks, onsets, mid_amps] = detector_name(sig, fs)
```

`detect_ppg_beats.m` calls any detector by name, making it trivial to add new algorithms. The `source/dev/` folder contains parameter-variant and experimental detectors used for publication-specific analyses.

ECG beat detectors are in `source/beat_detection_algorithms/ecg_beat_detection/`.

### Post-Processing

`tidy_beats.m` enforces structural rules on all detected beats (alternating peaks/onsets, no duplicates, valid timing). All detectors run through this after detection.

### Dataset Support

The toolbox benchmarks against multiple public datasets: CapnoBase, BIDMC, MIMIC PERform, PPG-DaLiA, WESAD. Dataset-specific loading functions follow a `collate_*` or `mimic_*` naming pattern in `source/`.

## Adding a New PPG Beat Detector

Per `CONTRIBUTING.md`, new detectors must:
1. Be placed in `source/` as `new_detector_beat_detector.m`
2. Follow the standard function signature above
3. Include the standard MATLAB comment block (author, date, license, description, inputs/outputs, references)
4. Use GPL-3.0 license (or compatible)
5. Have a corresponding `docs/functions/new_detector_beat_detector.md` documentation file

Improvements to existing algorithms should be a new file (not modifying the original) to preserve reproducibility of published results.
