
# Instructions for Calculation of OFDM Symbols

## 1. System Setup

- We have only one device, meaning this will be a point-to-point communication system.
- As a result, the number of subcarriers, denoted as `K`, is set to 1.

## 2. Simplifications

- The parameter `a`, as mentioned in the original paper, will be removed (this does not refer to the roll-off factor).
- The parameter `b` lies within the range `[0, sqrt(P)]` and can be determined using a Sigmoid DNN layer or by employing equalization:
  - If `1/h < sqrt(P)`, then `b = 1/h`.
  - Otherwise, `b = sqrt(P)`.

## 3. Dataset Details

The new dataset contains symbols from the QAM constellation family. These symbols are complex numbers calculated based on the following instructions:

### 3.1. Symbol Definition

- Let `C_j`, where `j ∈ J`, be the `j`-th frequency-domain OFDM constellation with order `M(j)`.
- Let `c(k, j) ∈ C_j` be a symbol taken from the `j`-th constellation, which will be transmitted on the `k`-th subcarrier.
- We assume that all `c(k, j)` are chosen uniformly at random from the `j`-th constellation `C_j`.

### 3.2. Symbol Coordinates

The coordinates of any symbol `c(k, j)` are given as follows:

- Real part:

  ```
  Re(c(k, j)) = (2ψ - 2D - 1) / Δ
  ```

- Imaginary part:

  ```
  Im(c(k, j)) = (2ζ - 2D - 1) / Δ
  ```

Where:

- `D = 2^(0.5 * log2(M_j / 4))`
- `ψ` and `ζ` are integers such that `1 ≤ ψ, ζ ≤ D`
- `Δ` is a normalization factor chosen such that `E[|c(k, j)|^2] = 1`.

### 3.3. Normalization Factor `Δ`

For a square `M`-QAM constellation, `Δ` is given by:

```
Δ = sqrt(2(M_j - 1) / 3)
```

## 4. Modified Mean Squared Error (MSE)

Since our new symbols `x` are now two values (Real and Imaginary parts) instead of one (which was between `-sqrt(3)` and `sqrt(3)`), the Mean Squared Error (MSE) will also change.

- The MSE will now involve two sums for the two components `x` and `y`.
