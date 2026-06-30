# Project Report: MNIST Digit Recognition & Overfitting Analysis

## 1. Objective
The goal of this project is to implement an image digit recognition system using the MNIST dataset, deliberately induce model overfitting using a high-capacity architecture, and subsequently mitigate that overfitting using training-based regularization techniques—all without altering the underlying model architecture.

## 2. Dataset & Data Splitting
The project utilizes the standard **MNIST dataset**, which consists of 70,000 grayscale images of handwritten digits (0-9). The data was rigorously split to ensure the model could be evaluated properly:
*   **Training Set:** 50,000 images (used to actively train the model).
*   **Validation Set:** 10,000 images (used to monitor overfitting during training).
*   **Test Set:** 10,000 images (used for the final, unbiased evaluation).

A fixed random seed was used during the splitting process to ensure identical datasets across all experiments.

## 3. Model Architecture (`LargeMLP`)
To successfully demonstrate overfitting on a relatively simple dataset like MNIST, a high-capacity Multi-Layer Perceptron (MLP) was constructed. The architecture consists of:
1.  **Input Layer:** Flattening the 28x28 image into a 1D array of 784 pixels.
2.  **Hidden Layer 1:** 1,024 neurons with a ReLU activation function.
3.  **Hidden Layer 2:** 1,024 neurons with a ReLU activation function.
4.  **Hidden Layer 3:** 512 neurons with a ReLU activation function.
5.  **Output Layer:** 10 neurons corresponding to the 10 digit classes.

This architecture contains roughly **1.8 million parameters**, granting it the "memory capacity" to easily memorize the training images if left unconstrained.

## 4. Phase 1: Inducing Overfitting
During Phase 1, the `LargeMLP` model was trained for 15 epochs using the Adam optimizer without any regularization.

**Results:**
*   The **Training Accuracy** reached near-perfect levels (~99.46%), and the Training Loss dropped close to 0.
*   However, the **Validation Loss** stopped improving around Epoch 5 and began to increase.
*   **Conclusion:** The divergence between the decreasing Training Loss and the increasing Validation Loss is the mathematical proof of overfitting. The model stopped learning general shapes and began purely memorizing the exact pixels of the training set.

## 5. Phase 2: Mitigating Overfitting (Regularization)
The challenge in Phase 2 was to overcome the overfitting problem without modifying the `LargeMLP` class (meaning structural changes like adding Dropout or Batch Normalization layers were strictly prohibited).

To achieve this, two training-based regularization techniques were implemented:
1.  **Data Augmentation:** Random rotations (up to 15 degrees) and random spatial translations (up to 10% shifts) were applied to the training images on the fly. Because the model never saw the exact same image twice, it was physically prevented from memorizing pixel layouts and was forced to learn the general shape of the digits.
2.  **Weight Decay (L2 Regularization):** A mathematical penalty (`weight_decay=1e-3`) was added to the Adam optimizer. This penalty forced the optimizer to keep the neural network's weights as small as possible, effectively restricting the model's complexity and forcing it to find simpler, more generalized rules.

**Results:**
*   While the **Training Accuracy** was lower compared to Phase 1 (dropping to ~96.1%), the **Validation Loss** dropped significantly (from `0.1204` down to `0.0650`).
*   **Conclusion:** The model successfully generalized to unseen data. By constraining the model's learning process, we prevented memorization and built a more robust digit recognizer.

## 6. Summary
This project successfully highlights the dangers of using unconstrained, high-capacity neural networks on simple datasets. Furthermore, it demonstrates that architectural changes are not strictly necessary to fix overfitting; training techniques such as Data Augmentation and Weight Decay are highly effective tools for forcing a model to generalize.
