# DRDO Internship — Machine Learning Adversarial Attacks

A literature-based study of adversarial attacks on machine learning systems, completed as part of an internship project at **DRDO** (Defence Research and Development Organisation). The work surveys how deep learning models can be fooled, how those attacks are classified, and how models can be defended and evaluated, with a particular focus on security-sensitive and defence applications.

> **Note:** This folder contains documentation only (a written report and a slide deck). There is no source code, dataset or trained model here.

---

## Contents

| File | Description | Size |
|------|-------------|------|
| [`drdo final.pdf`](./drdo%20final.pdf) | Full written report: *Machine Learning Adversarial Attacks: A Comprehensive Analysis of Threats, Defenses, and Robustness Evaluation* | 22 pages |
| [`ppt.pdf`](./ppt.pdf) | Internship evaluation presentation: *Machine Learning Adversarial Attacks — Understanding Security Vulnerabilities in AI Systems* | 12 slides |

---

## Project Overview

Deep neural networks perform very well in vision, language and autonomous systems, but they can be misled by small, deliberately crafted input perturbations that humans cannot notice. This project examines that weakness from both sides:

- **Threats:** how adversarial examples are built and under what assumptions (white-box, black-box, gray-box).
- **Defences:** how robustness is improved and what trade-offs each method carries.
- **Evaluation:** which metrics and protocols make robustness claims trustworthy.
- **Tooling:** how the IBM **Adversarial Robustness Toolkit (ART)** supports attack, defence and evaluation workflows.

### Problem statement

Models that score highly on clean data often degrade sharply on adversarially perturbed inputs. This accuracy-versus-robustness trade-off creates real risk in mission-critical systems where reliability and security matter.

---

## Report Structure (`drdo final.pdf`)

1. **Introduction** — problem statement and contributions
2. **Background and Literature Review** — formal definition of adversarial examples (bounded perturbation `‖x′ − x‖ₚ ≤ ε` with `f(x′) ≠ y`), history from Szegedy et al. and FGSM onwards
3. **Attack Taxonomies** — by adversary knowledge, by objective (targeted / untargeted), and by perturbation constraint (L∞, L2, L0, semantic)
4. **White-box Attacks** — FGSM, PGD, Carlini & Wagner, Auto-Attack, adaptive attacks
5. **Black-box Attacks** — query-based (zeroth-order, NES), transfer-based, score-based
6. **Defence Mechanisms** — adversarial training, defensive distillation, detection, input preprocessing, certified defences
7. **Adversarial Robustness Toolkit (ART)** — architecture, attack and defence catalogue, TensorFlow and PyTorch integration, extensibility, deployment, community impact
8. **Evaluation Methodologies** — robust accuracy, attack success rate, perturbation magnitude, benchmarks (MNIST, CIFAR-10/100, ImageNet), statistical rigour
9. **Applications and Case Studies** — autonomous systems, defence and security, medical AI, financial security
10. **Future Challenges and Conclusion**

---

## Presentation Outline (`ppt.pdf`)

| # | Slide |
|---|-------|
| 1 | Title |
| 2 | Contents |
| 3 | What are adversarial attacks? |
| 4 | Attack classification matrix (white-box / black-box, targeted / non-targeted) |
| 5 | Fast Gradient Sign Method (FGSM) |
| 6 | Attack success rates by model type |
| 7 | Attack generation pipeline |
| 8 | Real-world attack scenarios |
| 9 | Defence strategy framework |
| 10 | Defence effectiveness metrics |
| 11 | Future research directions |
| 12 | Thank you / Q&A |

---

## Key Concepts at a Glance

**Attack knowledge levels**

| Type | Attacker knowledge | Typical methods |
|------|--------------------|-----------------|
| White-box | Full: architecture, parameters, gradients | FGSM, PGD, C&W, Auto-Attack |
| Gray-box | Partial: e.g. architecture but not weights | — |
| Black-box | Inputs and outputs only | Zeroth-order optimisation, NES, transfer attacks |

**Core formulas covered**

- FGSM: `x_adv = x + ε · sign(∇ₓ J(θ, x, y))`
- PGD: iterated FGSM steps with projection onto the ε-ball
- Adversarial training: `min_θ E[ max_{‖δ‖≤ε} L(θ, x + δ, y) ]`

**Defence families**

| Defence | Idea | Trade-off noted in the slides |
|---------|------|-------------------------------|
| Adversarial training | Train on adversarial examples | Computationally expensive |
| Input transformation | Denoise, compress or randomise inputs | None listed |
| Detection methods | Flag suspicious inputs | May produce false positives |
| Certified defences | Mathematical robustness guarantees | Limited to small perturbations, high complexity |

---

## How to Use

Open either PDF directly in the GitHub viewer, or clone the repository and view locally:

```bash
git clone https://github.com/parul19012006/DL_Project.git
cd "DL_Project/DRDO Intern"
```

Suggested reading order: skim `ppt.pdf` for the overview, then read `drdo final.pdf` for depth.

---

## Limitations

- The report is a **survey and analysis**; it does not include original experiments or reproducible code.
- Figures in the slides (for example attack success rates by architecture, and defence effectiveness percentages) are presented as illustrative summaries without a stated dataset, model or experimental setup, so they should not be treated as measured results.
- The reference list in the report is shorter than its in-text citation range, so a few numbered citations do not map to a listed entry.

---

## References (as listed in the report)

1. Szegedy et al. (2013). *Intriguing properties of neural networks.* arXiv:1312.6199
2. Goodfellow, Shlens & Szegedy (2014). *Explaining and harnessing adversarial examples.* arXiv:1412.6572
3. Madry et al. (2017). *Towards deep learning models resistant to adversarial attacks.* arXiv:1706.06083
4. Carlini & Wagner (2017). *Towards evaluating the robustness of neural networks.* IEEE S&P

---

## Author

**Parul Gupta** — CSE (AI)
Internship project at DRDO
