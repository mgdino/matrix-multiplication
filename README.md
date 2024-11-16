# Matrix Multiplication Algorithms

This repository implements three methods for performing matrix multiplication:

1. **Naive Matrix Multiplication**
2. **Strassen's Algorithm**
3. **Coppersmith-Winograd Algorithm**

The code is written in Python and leverages NumPy for cleaner matrix handling.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
  - [Naive Matrix Multiplication](#naive-matrix-multiplication)
  - [Strassen's Algorithm](#strassens-algorithm)
  - [Coppersmith-Winograd Algorithm](#coppersmith-winograd-algorithm)



## Overview

This repositpry demonstrates different approaches to matrix multiplication, suitable for learning about algorithm design and optimization in numerical computing.


## Features

- **Matrix Multiplication Techniques**:
  - Naive approach for simplicity.
  - Strassen's algorithm for improved time complexity over naive multiplication.
  - Coppersmith-Winograd for advanced multiplication methods.
  
- **Matrix Splitting**:
  - Handles matrix splitting efficiently for Strassen's and Coppersmith-Winograd algorithms.

- **Error Checking**:
  - Validates the correctness of the multiplication results.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mgdino/matrix-multiplication.git
   cd matrix-multiplication
   pip install numpy
    ```

## Usage

1. Run the Python script to execute the matrix multiplication methods:

   ```bash
    python tester.py
    ```


## Algorithms

### Naive Matrix Multiplication

- **Time Complexity**: \(O(n^3)\)
- **Description**: 
  - Direct implementation of the mathematical definition of matrix multiplication.
  - Iterates through rows of the first matrix and columns of the second matrix.

### Strassen's Algorithm
- **Time Complexity**: \(O(n^{2.81})\)
- **Description**:
  - Uses divide-and-conquer to split matrices into submatrices.
  - Performs seven recursive multiplications instead of eight as in naive multiplication.

### Coppersmith-Winograd Algorithm
- **Time Complexity**: \(O(n^2.376)\)
- **Description**:
  - An advanced algorithm that reduces the number of multiplications required.
  - Simplified implementation based on this [paper](https://www-auth.cs.wisc.edu/lists/theory-reading/2009-December/pdfmN6UVeUiJ3.pdf)

