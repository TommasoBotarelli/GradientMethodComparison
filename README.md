# Gradient Descent Optimization: A Comparative Analysis of Line Search and Step-Size Strategies

## 🚀 Overview
This project evaluates the performance of the **Gradient Descent method** by benchmarking various **Line Search** techniques and **Initial Step Size** strategies. The goal was to identify high-efficiency configurations for solving large-scale non-linear optimization problems, focusing on the trade-offs between convergence speed and solution quality.

---

## 🛠️ Technologies & Methodologies

### 1. Advanced Line Search Algorithms
The study explores both **Monotone** and **Non-Monotone** strategies to determine the optimal step length ($\alpha_k$):
* **Armijo (Monotone):** A classical strategy ensuring a sufficient decrease in the objective function at every iteration.
* **Grippo-Lampariello-Lucidi (GLL):** A non-monotone variant that allows temporary increases in the objective function by comparing the current value against a recent maximum. This allows the algorithm to "escape" narrow valleys or local minima more effectively.
* **NLSA (Non-Monotone):** Replaces the maximization in GLL with a weighted average of previous objective values, providing a flexible degree of non-monotonicity.

### 2. Step-Size Initialization
To avoid the "slow-start" problem of constant steps, the project implements:
* **Barzilai-Borwein (BB1 & BB2):** Two-point step size estimators based on the secant equation, significantly accelerating convergence in high-dimensional spaces.
* **Constant Step (CS):** A baseline $\alpha_0 = 1$ used for rigorous performance comparison.

### 3. Benchmarking Environment
* **Language:** Python.
* **Dataset:** 15 diverse non-linear problems from the **PyCUTEst** library.
* **Scalability:** Tested on problems ranging from **2 to 10,000 dimensions** (e.g., ARWHEAD, BOX, SPARSQUR).
* **Competitor:** Benchmarked against the industry-standard **L-BFGS** solver from the SciPy library.

---

## 📈 Key Findings & Performance Advantages

### **Superior Convergence Speed**
The experimentation demonstrated that **Non-Monotone GLL (especially with window $M=10$)** consistently converges faster than monotone methods. By considering historical data, the algorithm avoids being trapped in restrictive descent requirements that often stall standard solvers.

###   **The Power of Intelligent Initialization**
Switching from a Constant Step (CS) to **Barzilai-Borwein (BB)** initialization resulted in speed increases of **several orders of magnitude**. 
* **BB2** not only reduced execution time but frequently led to **superior final objective values**.

### **Bespoke vs. General-Purpose Solvers**
Custom implementations often achieved comparable or **superior speeds** to the `L-BFGS` solver. This highlights the effectiveness of tailoring optimization strategies to specific problem classes rather than relying solely on black-box tools.

---

## 📋 Experimental Results Summary
The implementation successfully handled high-dimensional problems, demonstrating robust scalability:

| Problem Name | Dimensions | Complexity |
| :--- | :--- | :--- |
| **LOGHAIRY** | 2 | Low-dimensional |
| **SENSORS** | 100 | Mid-scale |
| **DIXMAANB** | 3,000 | Large-scale |
| **SPARSQUR** | 10,000 | Ultra-scale |

---

## 💡 Industry Applications
* **Large-Scale Machine Learning:** Efficiently optimizing loss functions in high-dimensional parameter spaces.
* **Engineering Design:** Navigating complex, non-convex surfaces where local minima are prevalent.
* **Performance Engineering:** Implementing low-level optimization routines that outperform standard library defaults.
