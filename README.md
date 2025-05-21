# Improved Physics-Informed Neural Network (I-PINN)

## Introduction
This repository contains the code and data for the research paper titled "Improved physics-informed neural network in mitigating gradient-related failures" published in Neurocomputing. The paper proposes an enhanced version of the Physics-Informed Neural Network (PINN), known as I-PINN, which addresses the issue of gradient flow stiffness and improves the predictive capabilities of PINNs.

## Paper Information
- **Title:** Improved physics-informed neural network in mitigating gradient-related failures
- **Authors:** Pancheng Niu, Jun Guo, Yongming Chen, Yuqian Zhou, Minfu Feng, Yanchao Shi
- **Journal:** Neurocomputing
- **Volume:** 638
- **Pages:** 130167
- **Year:** 2025
- **DOI:** [10.1016/j.neucom.2025.130167](https://doi.org/10.1016/j.neucom.2025.130167)
- **URL:** [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0925231225008392)
- **Keywords:** Physics-informed neural networks, Gradient flow stiffness, Adaptive weighting, Scientific computing

## Abstract
Physics-informed neural network (PINN) integrates fundamental physical principles with advanced data-driven techniques, leading to significant advancements in scientific computing. However, PINN encounters persistent challenges related to stiffness in gradient flow, which limits their predictive capabilities. This paper introduces an improved PINN (I-PINN) designed to mitigate gradient-related failures. The core of I-PINN combines the respective strengths of neural networks with an improved architecture and adaptive weights that include upper bounds. I-PINN achieves improved accuracy by at least one order of magnitude and accelerates convergence without introducing additional computational complexity compared to the baseline model. Numerical experiments across a variety of benchmarks demonstrate the enhanced accuracy and generalization of I-PINN. The supporting data and code are accessible at [GitHub Repository](https://github.com/PanChengN/I-PINN.git), facilitating broader research engagement.

## Citation
If you use this code or data in your research, please cite the following paper:
```bibtex
@article{NIU2025130167,
  title = {Improved physics-informed neural network in mitigating gradient-related failures},
  journal = {Neurocomputing},
  volume = {638},
  pages = {130167},
  year = {2025},
  issn = {0925-2312},
  doi = {10.1016/j.neucom.2025.130167},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231225008392},
  author = {Pancheng Niu and Jun Guo and Yongming Chen and Yuqian Zhou and Minfu Feng and Yanchao Shi},
  keywords = {Physics-informed neural networks, Gradient flow stiffness, Adaptive weighting, Scientific computing},
  abstract = {Physics-informed neural network (PINN) integrates fundamental physical principles with advanced data-driven techniques, leading to significant advancements in scientific computing. However, PINN encounters persistent challenges related to stiffness in gradient flow, which limits their predictive capabilities. This paper introduces an improved PINN (I-PINN) designed to mitigate gradient-related failures. The core of I-PINN combines the respective strengths of neural networks with an improved architecture and adaptive weights that include upper bounds. I-PINN achieves improved accuracy by at least one order of magnitude and accelerates convergence without introducing additional computational complexity compared to the baseline model. Numerical experiments across a variety of benchmarks demonstrate the enhanced accuracy and generalization of I-PINN.}
}
