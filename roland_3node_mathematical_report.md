# Mathematical Worked Example: ROLAND Dynamic GNN for EVCS Demand Forecasting
---

## 1. Worked Example Setup

The numerical weights used in this worked example are deterministic demonstration weights defined in the accompanying Python script (`roland_3node_worked_example.py`). They are intentionally small so that the matrix operations can be verified manually. They are not presented as the trained weights of the full production model.

We use three real stations from the corrected dataset so that the complete numerical computation can be followed manually. The stations belong to the **HYDERABAD** district in snapshot **`2024-06`**.

| Node | Node ID (`node_id`) | District | Units ($\text{kWh}$) | Load ($\text{kW}$) | Charger Age (months) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Node A** | `438` | `HYDERABAD` | $8,659.0$ | $328.0$ | $9$ |
| **Node B** | `486` | `HYDERABAD` | $94,531.0$ | $786.0$ | $5$ |
| **Node C** | `320` | `HYDERABAD` | $39,785.0$ | $1,073.0$ | $17$ |

---

## 2. Three Features Out of the Existing 56

The full project representation contains 56 features. For this mathematical demonstration, only three features are selected so that every operation can be calculated manually:
1. `units` (Index 44 in the full project vector)
2. `load` (Index 45 in the full project vector)
3. `charger_age_months` (Index 46 in the full project vector)

The raw observation vectors are:

$$\mathbf{x}_A(\text{raw}) = \begin{bmatrix} 8659.0 \\ 328.0 \\ 9.0 \end{bmatrix}, \quad \mathbf{x}_B(\text{raw}) = \begin{bmatrix} 94531.0 \\ 786.0 \\ 5.0 \end{bmatrix}, \quad \mathbf{x}_C(\text{raw}) = \begin{bmatrix} 39785.0 \\ 1073.0 \\ 17.0 \end{bmatrix}$$

---

## 3. Normalization

Normalization puts features with different physical scales into a comparable numerical range before they enter the neural network.

### 3.1 Normalization Formula
$$z = \frac{x - \mu}{\sigma}$$

Using the parameters fitted on active stations across the 26 training months from `dataset.py`:
$$\boldsymbol{\mu} = \begin{bmatrix} 12823.9473 \\ 435.5880 \\ 11.2521 \end{bmatrix}, \quad \boldsymbol{\sigma} = \begin{bmatrix} 19467.2871 \\ 1962.5702 \\ 7.4278 \end{bmatrix}$$

### 3.2 Step-by-Step Calculation for Node A
* **Feature 1 (`units`)**:
  $$z_{A, 1} = \frac{8659.0 - 12823.9473}{19467.2871} = \frac{-4164.9473}{19467.2871} = \mathbf{-0.213946}$$
* **Feature 2 (`load`)**:
  $$z_{A, 2} = \frac{328.0 - 435.5880}{1962.5702} = \frac{-107.5880}{1962.5702} = \mathbf{-0.054820}$$
* **Feature 3 (`charger_age_months`)**:
  $$z_{A, 3} = \frac{9.0 - 11.2521}{7.4278} = \frac{-2.2521}{7.4278} = \mathbf{-0.303199}$$

$$\mathbf{x}_A = \begin{bmatrix} -0.213946 \\ -0.054820 \\ -0.303199 \end{bmatrix}$$

### 3.3 Normalized Vectors for Nodes B and C
$$\mathbf{x}_B = \begin{bmatrix} \frac{94531.0 - 12823.9473}{19467.2871} \\ \frac{786.0 - 435.5880}{1962.5702} \\ \frac{5.0 - 11.2521}{7.4278} \end{bmatrix} = \begin{bmatrix} \mathbf{+4.197146} \\ \mathbf{+0.178547} \\ \mathbf{-0.841716} \end{bmatrix}$$

$$\mathbf{x}_C = \begin{bmatrix} \frac{39785.0 - 12823.9473}{19467.2871} \\ \frac{1073.0 - 435.5880}{1962.5702} \\ \frac{17.0 - 11.2521}{7.4278} \end{bmatrix} = \begin{bmatrix} \mathbf{+1.384941} \\ \mathbf{+0.324784} \\ \mathbf{+0.773836} \end{bmatrix}$$

---

## 4. Graph Construction & Degree Normalization

In the project graph construction, stations in the same district form a complete subgraph. The GNN convolution adds self-loops.

### 4.1 Adjacency Matrix with Self-Loops ($\mathbf{A}$)
$$\mathbf{A} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

### 4.2 Degree Matrix ($\mathbf{D}$) and Inverse Square Root ($\mathbf{D}^{-1/2}$)
Each node has degree $d_i = 1 + 1 + 1 = 3$:
$$\mathbf{D} = \begin{bmatrix} 3 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 3 \end{bmatrix}, \quad \mathbf{D}^{-1/2} = \begin{bmatrix} \frac{1}{\sqrt{3}} & 0 & 0 \\ 0 & \frac{1}{\sqrt{3}} & 0 \\ 0 & 0 & \frac{1}{\sqrt{3}} \end{bmatrix}$$

### 4.3 Symmetric Normalized Adjacency ($\widetilde{\mathbf{A}}$)
$$\widetilde{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} = \begin{bmatrix} 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \end{bmatrix} \approx \begin{bmatrix} 0.333333 & 0.333333 & 0.333333 \\ 0.333333 & 0.333333 & 0.333333 \\ 0.333333 & 0.333333 & 0.333333 \end{bmatrix}$$

The normalized adjacency determines how much each incoming message contributes during aggregation.

---

## 5. ResidualEdgeConv

For an edge from source $j$ to target $i$, the layer constructs:
$$\mathbf{x}_{\text{edge}} = [\mathbf{x}_i \,\|\, \mathbf{x}_j] \in \mathbb{R}^6$$

The message function receives both the target and source representations, allowing the learned message to depend on the interaction between the two stations rather than only on the neighbor.

### 5.1 Demonstration Message MLP

$$\mathbf{m}_{ij} = \mathbf{W}_{\text{msg2}} \cdot \text{ReLU}\left(\mathbf{W}_{\text{msg1}} [\mathbf{x}_i \,\|\, \mathbf{x}_j] + \mathbf{b}_{\text{msg1}}\right)$$

The real project model has larger hidden dimensions. The dimensions are reduced here only so that the complete arithmetic can be checked manually. The sequence of operations is unchanged: $\text{Linear}(6 \to 4) \to \text{ReLU} \to \text{Linear}(4 \to 2)$.

#### Parameters from Python Script:
$$\mathbf{W}_{\text{msg1}} = \begin{bmatrix} 0.2 & -0.1 & 0.3 & 0.1 & 0.4 & -0.2 \\ -0.3 & 0.2 & 0.1 & -0.2 & 0.1 & 0.3 \\ 0.1 & 0.3 & -0.2 & 0.3 & -0.1 & 0.2 \\ -0.2 & 0.1 & 0.4 & -0.1 & 0.2 & 0.1 \end{bmatrix}, \quad \mathbf{b}_{\text{msg1}} = \begin{bmatrix} +0.10 \\ -0.05 \\ +0.05 \\ 0.00 \end{bmatrix}$$

$$\mathbf{W}_{\text{msg2}} = \begin{bmatrix} 0.4 & -0.3 & 0.2 & 0.5 \\ -0.2 & 0.5 & 0.1 & -0.4 \end{bmatrix}$$

---

### 5.2 Step-by-Step Message Calculation for Edge $B \to A$ ($\mathbf{m}_{AB}$)

#### Step 1: Concatenation $[\mathbf{x}_A \,\|\, \mathbf{x}_B]$
$$\mathbf{x}_{\text{edge}, AB} = \begin{bmatrix} -0.213946 \\ -0.054820 \\ -0.303199 \\ +4.197146 \\ +0.178547 \\ -0.841716 \end{bmatrix} \in \mathbb{R}^6$$

#### Step 2: First Linear Transformation ($\mathbf{u} = \mathbf{W}_{\text{msg1}} \mathbf{x}_{\text{edge}, AB} + \mathbf{b}_{\text{msg1}}$)
* $u_1 = 0.2(-0.213946) - 0.1(-0.054820) + 0.3(-0.303199) + 0.1(4.197146) + 0.4(0.178547) - 0.2(-0.841716) + 0.10 = \mathbf{+0.702580}$
* $u_2 = -0.3(-0.213946) + 0.2(-0.054820) + 0.1(-0.303199) - 0.2(4.197146) + 0.1(0.178547) + 0.3(-0.841716) - 0.05 = \mathbf{-1.100913}$
* $u_3 = 0.1(-0.213946) + 0.3(-0.054820) - 0.2(-0.303199) + 0.3(4.197146) - 0.1(0.178547) + 0.2(-0.841716) + 0.05 = \mathbf{+1.151745}$
* $u_4 = -0.2(-0.213946) + 0.1(-0.054820) + 0.4(-0.303199) - 0.1(4.197146) + 0.2(0.178547) + 0.1(-0.841716) + 0.00 = \mathbf{-0.551065}$

$$\mathbf{u} = \begin{bmatrix} +0.702580 \\ -1.100913 \\ +1.151745 \\ -0.551065 \end{bmatrix}$$

#### Step 3: ReLU Activation ($\mathbf{h} = \max(0, \mathbf{u})$)
$$\mathbf{h} = \begin{bmatrix} \max(0, +0.702580) \\ \max(0, -1.100913) \\ \max(0, +1.151745) \\ \max(0, -0.551065) \end{bmatrix} = \begin{bmatrix} 0.702580 \\ 0.000000 \\ 1.151745 \\ 0.000000 \end{bmatrix}$$

#### Step 4: Second Linear Transformation ($\mathbf{m}_{AB} = \mathbf{W}_{\text{msg2}} \mathbf{h}$)
* $m_{AB, 1} = 0.4(0.702580) - 0.3(0) + 0.2(1.151745) + 0.5(0) = 0.281032 + 0.230349 = \mathbf{+0.481633}$
* $m_{AB, 2} = -0.2(0.702580) + 0.5(0) + 0.1(1.151745) - 0.4(0) = -0.140516 + 0.115175 = \mathbf{-0.011668}$

$$\mathbf{m}_{AB} = \begin{bmatrix} +0.481633 \\ -0.011668 \end{bmatrix}$$

---

### 5.3 All 9 Directional Message Vectors

Applying the same operation across all edges yields:

| Message | Source $\to$ Target | Resulting Vector $\mathbf{m}_{ij}$ |
|:---:|:---:|:---:|
| $\mathbf{m}_{AA}$ | $A \to A$ (Self-loop) | $[0.000000, 0.000000]^T$ |
| $\mathbf{m}_{AB}$ | $B \to A$ | $[+0.481633, -0.011668]^T$ |
| $\mathbf{m}_{AC}$ | $C \to A$ | $[+0.156264, +0.043982]^T$ |
| $\mathbf{m}_{BA}$ | $A \to B$ | $[+0.389007, -0.080047]^T$ |
| $\mathbf{m}_{BB}$ | $B \to B$ (Self-loop) | $[+0.884328, -0.089251]^T$ |
| $\mathbf{m}_{BC}$ | $C \to B$ | $[+0.558959, -0.033601]^T$ |
| $\mathbf{m}_{CA}$ | $A \to C$ | $[+0.262523, -0.135668]^T$ |
| $\mathbf{m}_{CB}$ | $B \to C$ | $[+0.735277, -0.126817]^T$ |
| $\mathbf{m}_{CC}$ | $C \to C$ (Self-loop) | $[+0.444343, -0.098715]^T$ |

---

## 6. Aggregation + Skip Connection

### 6.1 Aggregation for Node A
$$\mathbf{aggr}_A = \frac{1}{3}\mathbf{m}_{AA} + \frac{1}{3}\mathbf{m}_{AB} + \frac{1}{3}\mathbf{m}_{AC}$$

$$\mathbf{aggr}_A = \frac{1}{3}\left(\begin{bmatrix} 0.000000 \\ 0.000000 \end{bmatrix} + \begin{bmatrix} +0.481633 \\ -0.011668 \end{bmatrix} + \begin{bmatrix} +0.156264 \\ +0.043982 \end{bmatrix}\right) = \frac{1}{3}\begin{bmatrix} +0.637897 \\ +0.032314 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.212632} \\ \mathbf{+0.010772} \end{bmatrix}$$

---

### 6.2 Skip Connection for Node A
The skip path preserves a direct transformation of the node's own features.

$$\mathbf{skip}_A = \mathbf{W}_{\text{skip}} \mathbf{x}_A + \mathbf{b}_{\text{skip}}$$

Using weights from the code:
$$\mathbf{W}_{\text{skip}} = \begin{bmatrix} 0.3 & 0.1 & -0.2 \\ -0.1 & 0.4 & 0.3 \end{bmatrix}, \quad \mathbf{b}_{\text{skip}} = \begin{bmatrix} +0.05 \\ -0.05 \end{bmatrix}$$

* $\text{skip}_{A, 1} = 0.3(-0.213946) + 0.1(-0.054820) - 0.2(-0.303199) + 0.05 = \mathbf{+0.040974}$
* $\text{skip}_{A, 2} = -0.1(-0.213946) + 0.4(-0.054820) + 0.3(-0.303199) - 0.05 = \mathbf{-0.141493}$

$$\mathbf{skip}_A = \begin{bmatrix} +0.040974 \\ -0.141493 \end{bmatrix}$$

---

### 6.3 Spatial Node Embedding for Node A ($\mathbf{X}_{t, A}$)
With conv layer bias $\mathbf{b}_{\text{conv}} = [0, 0]^T$:
$$\mathbf{X}_{t, A} = \mathbf{aggr}_A + \mathbf{skip}_A + \mathbf{b}_{\text{conv}} = \begin{bmatrix} +0.212632 \\ +0.010772 \end{bmatrix} + \begin{bmatrix} +0.040974 \\ -0.141493 \end{bmatrix} = \begin{bmatrix} \mathbf{+0.253606} \\ \mathbf{-0.130721} \end{bmatrix}$$

---

### 6.4 Summary Table for Nodes A, B, and C

| Node | Aggregation ($\mathbf{aggr}_i$) | Skip Path ($\mathbf{skip}_i$) | Spatial Embedding ($\mathbf{X}_{t, i}$) |
|:---:|:---:|:---:|:---:|
| **Node A** | $[+0.212632, +0.010772]^T$ | $[+0.040974, -0.141493]^T$ | $[+0.253606, -0.130721]^T$ |
| **Node B** | $[+0.610765, -0.067633]^T$ | $[+1.495342, -0.650811]^T$ | $[+2.106107, -0.718443]^T$ |
| **Node C** | $[+0.480714, -0.120400]^T$ | $[+0.343194, +0.173570]^T$ | $[+0.823908, +0.053171]^T$ |

---

## 7. GRU Temporal Update

The spatial GNN captures the current graph structure, while the GRU carries information from the previous time step into the current node state.

The miniature worked example uses hidden state dimension = 2.

### 7.1 Parameters from Python Script
$$\mathbf{W}_z = \begin{bmatrix} 0.3 & -0.2 & 0.4 & 0.1 \\ -0.1 & 0.5 & -0.2 & 0.3 \end{bmatrix}, \quad \mathbf{b}_z = \begin{bmatrix} +0.10 \\ -0.10 \end{bmatrix}$$

$$\mathbf{W}_r = \begin{bmatrix} 0.2 & 0.4 & -0.1 & 0.3 \\ -0.3 & 0.1 & 0.5 & -0.2 \end{bmatrix}, \quad \mathbf{b}_r = \begin{bmatrix} 0.00 \\ +0.10 \end{bmatrix}$$

$$\mathbf{W}_h = \begin{bmatrix} 0.5 & -0.1 & 0.3 & -0.2 \\ -0.2 & 0.4 & 0.1 & 0.5 \end{bmatrix}, \quad \mathbf{b}_h = \begin{bmatrix} +0.05 \\ -0.05 \end{bmatrix}$$

---

### 7.2 Complete Calculation for Node A

Inputs for Node A:
$$\mathbf{H}_{t-1, A} = \begin{bmatrix} +0.100000 \\ -0.100000 \end{bmatrix}, \quad \mathbf{X}_{t, A} = \begin{bmatrix} +0.253606 \\ -0.130721 \end{bmatrix}, \quad \mathbf{c}_A = [\mathbf{X}_{t, A} \,\|\, \mathbf{H}_{t-1, A}] = \begin{bmatrix} +0.253606 \\ -0.130721 \\ +0.100000 \\ -0.100000 \end{bmatrix}$$

#### 1. Update Gate ($\mathbf{Z}_t = \sigma(\mathbf{W}_z \mathbf{c}_A + \mathbf{b}_z)$)
* $a_{z1} = 0.3(0.253606) - 0.2(-0.130721) + 0.4(0.10) + 0.1(-0.10) + 0.10 = +0.232226 \implies Z_{t, 1} = \sigma(0.232226) = \mathbf{0.557797}$
* $a_{z2} = -0.1(0.253606) + 0.5(-0.130721) - 0.2(0.10) + 0.3(-0.10) - 0.10 = -0.240721 \implies Z_{t, 2} = \sigma(-0.240721) = \mathbf{0.440109}$

$$\mathbf{Z}_t = \begin{bmatrix} 0.557797 \\ 0.440109 \end{bmatrix}$$

#### 2. Reset Gate ($\mathbf{R}_t = \sigma(\mathbf{W}_r \mathbf{c}_A + \mathbf{b}_r)$)
* $a_{r1} = 0.2(0.253606) + 0.4(-0.130721) - 0.1(0.10) + 0.3(-0.10) + 0.00 = -0.041567 \implies R_{t, 1} = \sigma(-0.041567) = \mathbf{0.489610}$
* $a_{r2} = -0.3(0.253606) + 0.1(-0.130721) + 0.5(0.10) - 0.2(-0.10) + 0.10 = +0.080846 \implies R_{t, 2} = \sigma(0.080846) = \mathbf{0.520200}$

$$\mathbf{R}_t = \begin{bmatrix} 0.489610 \\ 0.520200 \end{bmatrix}$$

#### 3. Reset Memory ($\mathbf{r}_{\text{mem}} = \mathbf{R}_t \odot \mathbf{H}_{t-1, A}$)
$$\mathbf{r}_{\text{mem}} = \begin{bmatrix} 0.489610 \times 0.10 \\ 0.520200 \times (-0.10) \end{bmatrix} = \begin{bmatrix} +0.048961 \\ -0.052020 \end{bmatrix}$$

#### 4. Candidate State ($\widetilde{\mathbf{H}}_t = \tanh(\mathbf{W}_h [\mathbf{X}_{t, A} \,\|\, \mathbf{r}_{\text{mem}}] + \mathbf{b}_h)$)
* $a_{h1} = 0.5(0.253606) - 0.1(-0.130721) + 0.3(0.048961) - 0.2(-0.052020) + 0.05 = +0.214954 \implies \widetilde{H}_{t, 1} = \tanh(0.214954) = \mathbf{+0.211716}$
* $a_{h2} = -0.2(0.253606) + 0.4(-0.130721) + 0.1(0.048961) + 0.5(-0.052020) - 0.05 = -0.174116 \implies \widetilde{H}_{t, 2} = \tanh(-0.174116) = \mathbf{-0.172385}$

$$\widetilde{\mathbf{H}}_t = \begin{bmatrix} +0.211716 \\ -0.172385 \end{bmatrix}$$

#### 5. Updated State ($\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1, A} + (1 - \mathbf{Z}_t) \odot \widetilde{\mathbf{H}}_t$)
* $H_{t, 1} = 0.557797(0.10) + (1 - 0.557797)(0.211716) = \mathbf{+0.149401}$
* $H_{t, 2} = 0.440109(-0.10) + (1 - 0.440109)(-0.172385) = \mathbf{-0.140528}$

$$\mathbf{H}_{t, A} = \begin{bmatrix} +0.149401 \\ -0.140528 \end{bmatrix}$$

---

### 7.3 Summary Table for All Nodes

| Node | Prior Memory ($\mathbf{H}_{t-1}$) | Update Gate ($\mathbf{Z}_t$) | Reset Gate ($\mathbf{R}_t$) | Candidate ($\widetilde{\mathbf{H}}_t$) | Updated Memory ($\mathbf{H}_t$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Node A** | $[+0.100000, -0.100000]^T$ | $[0.557797, 0.440109]^T$ | $[0.489610, 0.520200]^T$ | $[+0.211716, -0.172385]^T$ | $[+0.149401, -0.140528]^T$ |
| **Node B** | $[+0.250000, +0.050000]^T$ | $[0.727212, 0.330742]^T$ | $[0.530921, 0.380205]^T$ | $[+0.836954, -0.626613]^T$ | $[+0.410114, -0.402829]^T$ |
| **Node C** | $[-0.150000, +0.200000]^T$ | $[0.573596, 0.483555]^T$ | $[0.564894, 0.436137]^T$ | $[+0.391670, -0.157062]^T$ | $[+0.080970, +0.015597]^T$ |

---

## 8. Regression Head

The hidden dimension is reduced for manual calculation; the mathematical operation remains $\text{Linear} \to \text{ReLU} \to \text{Linear}$.

$$\hat{\mathbf{y}}_{\text{norm}} = \mathbf{W}_{\text{reg2}} \cdot \text{ReLU}\left(\mathbf{W}_{\text{reg1}} \mathbf{H}_t + \mathbf{b}_{\text{reg1}}\right) + \mathbf{b}_{\text{reg2}}$$

### 8.1 Parameters from Python Script
$$\mathbf{W}_{\text{reg1}} = \begin{bmatrix} 0.4 & -0.2 \\ -0.3 & 0.5 \end{bmatrix}, \quad \mathbf{b}_{\text{reg1}} = \begin{bmatrix} +0.10 \\ +0.05 \end{bmatrix}, \quad \mathbf{W}_{\text{reg2}} = \begin{bmatrix} 0.6 & -0.4 \\ -0.1 & 0.7 \end{bmatrix}, \quad \mathbf{b}_{\text{reg2}} = \begin{bmatrix} -0.20 \\ +0.10 \end{bmatrix}$$

---

### 8.2 Calculation for Node A

#### Layer 1 (Linear + ReLU):
* $v_1 = 0.4(0.149401) - 0.2(-0.140528) + 0.10 = +0.187866 \implies h_{\text{reg}, 1} = \max(0, 0.187866) = \mathbf{0.187866}$
* $v_2 = -0.3(0.149401) + 0.5(-0.140528) + 0.05 = -0.065084 \implies h_{\text{reg}, 2} = \max(0, -0.065084) = \mathbf{0.000000}$

$$\mathbf{h}_{\text{reg1}, A} = \begin{bmatrix} 0.187866 \\ 0.000000 \end{bmatrix}$$

#### Layer 2 (Output Projection):
* $\hat{y}_{\text{norm}, 1} = 0.6(0.187866) - 0.4(0) - 0.20 = \mathbf{-0.087280}$
* $\hat{y}_{\text{norm}, 2} = -0.1(0.187866) + 0.7(0) + 0.10 = \mathbf{+0.081213}$

$$\hat{\mathbf{y}}_{\text{norm}, A} = \begin{bmatrix} -0.087280 \\ +0.081213 \end{bmatrix}$$

---

### 8.3 Summary of Normalized Predictions

| Node | Hidden Activation ($\mathbf{h}_{\text{reg1}}$) | Normalized Prediction ($\hat{\mathbf{y}}_{\text{norm}}$) |
|:---:|:---:|:---:|
| **Node A** | $[0.187866, 0.000000]^T$ | $[-0.087280, +0.081213]^T$ |
| **Node B** | $[0.344611, 0.000000]^T$ | $[+0.006767, +0.065539]^T$ |
| **Node C** | $[0.129269, 0.033507]^T$ | $[-0.135842, +0.110528]^T$ |

---

## 9. Inverse Scaling & Final Numerical Output

Predictions are scaled back to physical units using the dataset mean and standard deviation:
$$\widehat{\text{Units}} = \hat{y}_{\text{norm}, 0} \times \sigma_{\text{units}} + \mu_{\text{units}}$$
$$\widehat{\text{Load}} = \hat{y}_{\text{norm}, 1} \times \sigma_{\text{load}} + \mu_{\text{load}}$$

### 9.1 Calculation for Node A
$$\widehat{\text{Units}}_A = (-0.087280) \times 19467.2871 + 12823.9473 = \mathbf{11,124.84\text{ kWh}}$$
$$\widehat{\text{Load}}_A = (+0.081213) \times 1962.5702 + 435.5880 = \mathbf{594.97\text{ kW}}$$

---

### 9.2 Final Outputs for All Three Nodes

| Node | Node ID (`node_id`) | Predicted Units ($\text{kWh}$) | Predicted Load ($\text{kW}$) |
|:---:|:---:|:---:|:---:|
| **Node A** | `438` | $11,124.84$ | $594.97$ |
| **Node B** | `486` | $12,955.68$ | $564.21$ |
| **Node C** | `320` | $10,179.48$ | $652.51$ |

These numerical outputs are generated using the miniature computational pipeline and demonstration weights to illustrate the end-to-end mathematical flow. They are not model-performance evaluations.

---

## 10. Compact Flow Diagram & Dimension Summary

### 10.1 Mathematical Flow Diagram

```
3 Real EVCS Nodes (A, B, C)
            │
            ▼
3 Selected Features (Units, Load, Age)
            │
            ▼
Z-Score Normalization (z = (x - mu) / sigma)
            │
            ▼
3-Node Graph (Complete Subgraph with Self-Loops)
            │
            ▼
Normalized Adjacency (A_tilde = D^(-1/2) A D^(-1/2))
            │
            ▼
Edge Concatenation ([x_i || x_j])
            │
            ▼
Message MLP (Linear -> ReLU -> Linear)
            │
            ▼
Normalized Aggregation (aggr_i = sum_j A_tilde_ij * m_ij)
            │
            ▼
Skip Connection (skip_i = W_skip * x_i + b_skip)
            │
            ▼
Spatial Embedding (X_t,i = aggr_i + skip_i + b_conv)
            │
            ▼
GRU Temporal Update (Z_t, R_t, H_tilde -> H_t)
            │
            ▼
Regression Head (Linear -> ReLU -> Linear)
            │
            ▼
Inverse Scaling -> Final Forecasts (Units, Load)
```

---

### 10.2 Dimension Trace Table

| Stage | Input Dimension | Output Dimension | Description |
|---|:---:|:---:|---|
| **Node Features** | $3$ | $3$ | Selected raw features (`units`, `load`, `charger_age`) |
| **Normalization** | $3$ | $3$ | Z-score standardization |
| **Edge Concatenation** | $3 + 3$ | $6$ | Concatenation $[\mathbf{x}_i \,\|\, \mathbf{x}_j]$ |
| **Message MLP** | $6$ | $4 \to 2$ | Directional edge message $\mathbf{m}_{ij}$ |
| **Aggregation** | $2$ | $2$ | Neighborhood message sum $\sum_j \widetilde{A}_{ij} \mathbf{m}_{ij}$ |
| **Skip Path** | $3$ | $2$ | Affine projection $\mathbf{W}_{\text{skip}} \mathbf{x}_i + \mathbf{b}_{\text{skip}}$ |
| **Spatial Embedding ($\mathbf{X}_t$)** | $2$ | $2$ | Sum $\mathbf{aggr} + \mathbf{skip} + \mathbf{b}_{\text{conv}}$ |
| **GRU Temporal Update** | $2 + 2$ | $2$ | Recurrent memory update $\text{GRU}(\mathbf{X}_t, \mathbf{H}_{t-1})$ |
| **Regression Head** | $2$ | $2 \to 2$ | MLP mapping to normalized targets |
| **Output Scaling** | $2$ | $2$ | Inverse scaling to physical $\text{kWh}$ and $\text{kW}$ |

*(Dimensions used in the miniature worked example).*
