# Learning Robust Neural Networks Using Constraint Programming  

## Overview  
This project explores a novel neuro-symbolic approach to **training robust binarized neural networks (BNNs)** using **Constraint Programming**.  
While traditional training methods provide no formal guarantees of robustness against adversarial perturbations, this work uses CP to **enforce robustness and fairness constraints** directly into the training process.  

Building on the CP formulation of neural networks proposed by *Icarte et al. (2019)*, this framework aims to improve generalization, adversarial robustness, and fairness by modeling neural network training as a Constraint Optimization Problem.  

---

## Key Features  
- **Constraint Programming Training**: Formulates BNN training as a COP using IBM ILOG CP Optimizer.  
- **Robustness Enforcement**: Ensures adversarial robustness by incorporating robustness constraints based on defined perturbation set.  
- **Fairness Constraints**: Added as robustness with respect to sensitive attribute (fairness-through-unawareness).  
- **Dual Training Approaches**:  
  - Baseline BNN training with **Keras + Larq**  
  - CP-based BNN training with **Docplex + CPO**  
- **Evaluation on Real Datasets**: Tested on **German Credit** and **COMPAS** datasets, along with experiments on **MNIST** for comparison.  

---

## CP Formulation  

The CP model can be defined with different objectives:  

$$
\text{(Min weight:)} \quad \min \sum_{\ell \in L}\sum_{i \in N_{\ell -1}}\sum_{j \in N_{\ell}} |w_{ilj}|
$$

OR  

$$
\text{(Max margin:)} \quad \max \sum_{\ell \in L}\sum_{j \in N_{\ell}} \min \big(|w_{\ell j} \cdot n_{\ell -1}^{k}|, \, k \in \mathcal{T}\big)
$$

OR  

$$
\text{(Max Training Accuracy:)} \quad \max \sum_{k=1}^{|\mathcal{T}|} c_{k}
$$

subject to:  

$$
n_{0j}^{k} = x_{j}^{k}, \quad \forall j \in N_{0},\, k \in \mathcal{T}
$$  

$$
n_{\ell j}^{k} = 2(w_{\ell j}\cdot n_{\ell -1}^{k} \geq 0) - 1, \quad \forall \ell \in \{1,\ldots,L\}, \, j \in N_{\ell},\, k \in \mathcal{T}
$$  

$$
\rho |\mathcal{T}| \leq \sum_{k=1}^{|\mathcal{T}|}c_{k} < |\mathcal{T}|, \quad 0<\rho<1
$$  

where, $\rho$ represents the fraction of training examples correctly classified by the Keras model.  

---

## Robustness Constraints  

For robustness, we define perturbation sets and enforce neuron activation stability.  

For each sample $x$ and neuron $j$ in layer 1:  

$$
x_{j}^{up}[i] =
\begin{cases}
x[i]-\epsilon, & w_{i1j}\geq 0 \\
x[i]+\epsilon, & w_{i1j}<0
\end{cases}
$$

$$
x_{j}^{down}[i] =
\begin{cases}
x[i]+\epsilon, & w_{i1j}\geq 0 \\
x[i]-\epsilon, & w_{i1j}<0
\end{cases}
$$

Constraint:  
- If neuron $j$ is activated for sample $x$ and $x_{j}^{up}$, it is activated $\forall y \in S_{\epsilon}(x)$.  
- If neuron $j$ is not activated for $x$ and $x_{j}^{down}$, then it is not activated $\forall y \in S_{\epsilon}(x)$.  

---

## Tech Stack  
- **Languages**: Python 3.8  
- **Libraries**:  
  - Machine Learning: `Keras`, `TensorFlow`, `Larq`, `Scikit-learn`  
  - Optimization: `Docplex` 
  - Data Processing: `Pandas`, `NumPy`, `Imbalanced-learn`  
  - Visualization: `Seaborn`
- **Software**: IBM ILOG CP Optimizer  

---

## Datasets  
- **German Credit Dataset** → Sensitive attribute: *Age*  
- **Mushroom Dataset** → No sensitive attribute   

---


