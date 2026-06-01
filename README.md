<h2 align="center">
    Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling
</h2>

<h5 align="center">
    Seojeong Park<sup>*</sup>, Jiho Choi<sup>*</sup>, Junyong Kang, Seonho Lee, Jaeyo Shin, Hyunjung Shim<sup>†</sup><br/>
    <br/>
    <!-- <p>
        * equal contribution  † corresponding author
    </p> -->
    Graduate School of Artificial Intelligence, KAIST, Republic of Korea<br/>
    <!-- KRAFTON, Republic of Korea<br/> -->
    <br/>
    <!-- <code>{seojeong.park, jihochoi, kateshim}@kaist.ac.kr</code> -->
</h5>

<h4 align="center">
    <a href="https://perception-judge.github.io/"> <img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"> </a>
    <a href="#"> <img src="https://img.shields.io/badge/arXiv-TBA-b31b1b.svg" alt="arXiv"> </a>
</h4>

<div align="center">
    <img src="assets/2026_ICML.png" alt="teaser" width="90%"/>
</div>

<br/>

## Overview

**Perception-Judge** is a multimodal evaluator that reinforces *perceptual grounding* in
LLM-as-a-Judge. We identify and formalize **Perceptual Judgment Bias**, a systematic failure mode
in which multimodal LLM judges reward linguistically plausible yet visually ungrounded responses
when visual evidence conflicts with textual cues. To mitigate this bias, we construct the
**Perceptually Perturbed Judgment Dataset (PPJD)**, which applies controlled perceptual
perturbations to build minimally edited counterfactual responses that isolate perceptual errors and
enable verifiable supervision, and train the judge with a **GRPO-based verifiable batch-ranking
reward** that enforces perceptual verification as a prerequisite for reasoning, achieving coherent
global ordering without explicit pairwise labels.

<br/>

## Key Features

- 🔍 **Perceptual Judgment Bias**: A formal analysis decomposing judge errors into *insufficient perceptual capability* and *response anchoring*
- 🧪 **PPJD**: A perceptually perturbed judgment dataset that disentangles perceptual failures from reasoning errors via verifiable, counterfactual supervision
- 🏆 **Verifiable Batch-Ranking Reward**: A GRPO-based objective that induces globally consistent ranking without explicit pairwise labels

<br/>

## Updates

- 💻 **Code Release**: TBA

<br/>

<!--
## Installation

```bash
# TBA
```

<br/>

## Data Preparation

```bash
# TBA
```

<br/>

## Usage

```bash
# TBA
```

<br/>

## Evaluation

```bash
# TBA
```

<br/>
-->

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{perceptionjudge2026,
  title={Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling},
  author={Park, Seojeong and Choi, Jiho and Kang, Junyong and Lee, Seonho and Shin, Jaeyo and Shim, Hyunjung},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```
