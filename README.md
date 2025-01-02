# Circle2

## Overview
This project is an extention of the Circle experiment. The goal is to predict the parameters of four circles derived from five random points on a plane using a machine learning model.

---

## Key Steps

### 1. Dataset Preparation

#### Input Features
The input consists of 10 values representing the x and y coordinates of 5 points:
```plaintext
Input: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
```

#### Output Features
The output contains 12 values for the x and y coordinates of the centers and radii of 4 circles:
```plaintext
Output: [xc1, yc1, r1, xc2, yc2, r2, xc3, yc3, r3, xc4, yc4, r4]
```

#### Labeling
A dataset is generated programmatically using the following steps:
1. Randomly generate 5 points on a plane.
2. Compute all combinations of 3 points to form circles.
3. Check conditions:
   - One remaining point is inside the circle.
   - One remaining point is outside the circle.
4. Store valid circle parameters as outputs.

The circles are sorted based on their radii to maintain consistent order.

---

### 2. Model Architecture

A fully connected neural network is used.

#### Architecture
- **Input Layer**: 10 neurons (corresponding to the x, y coordinates of the points).
- **Hidden Layers**: 2–3 layers with 128–256 neurons each.
- **Output Layer**: 12 neurons (corresponding to the x, y coordinates of the centers and radii of the 4 circles).

---

### 3. Training Process

- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam optimizer.
- **Evaluation Metric**: Accuracy of predicted circle parameters compared to the ground truth.

#### Note
To ensure consistent ordering, the four circles are sorted by radius during dataset generation and after prediction.