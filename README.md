# 🧠 Neural Network Trainer for Predicting Movement Data

This project implements a simple feedforward neural network from scratch in Python, used to predict velocity (`vx`, `vy`) based on position data (`x`, `y`) from a game/simulation. It includes preprocessing, training, validation, and RMSE tracking.

---

## 🗕️ Date
**25/11/2024**

---

## 🔧 Features

- ✅ Custom feedforward neural network from scratch (no deep learning libraries)
- ✅ Support for momentum-based training
- ✅ Early stopping to prevent overfitting
- ✅ Outlier detection using Z-Score
- ✅ Min-Max normalization
- ✅ Optional bias in layers
- ✅ Plots training and validation RMSE
- ✅ Debug-friendly code with clear variable usage and comments

---

## 📁 Input

CSV File: `ce889_dataCollection(baha).csv`

Expected Columns:

| x | y | vx | vy |
|---|---|----|----|

- `x`, `y` → Input positions  
- `vx`, `vy` → Output velocities (target values)

---

## 📦 Requirements

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## ▶️ How to Run

Run the script and input the number of training epochs:

```bash
python neural_network_trainer.py
```

---

## 🧠 Model Architecture

- **Input Layer**: 2 neurons + bias
- **Hidden Layer**: 15 neurons (adjustable)
- **Output Layer**: 2 neurons (vx, vy)

Bias is included by default; it can be removed by commenting the relevant code lines.

---

## 📊 Training & Validation

- Training and validation are both monitored via RMSE (Root Mean Square Error).
- Early stopping is applied if no significant improvement is seen in RMSE over a few epochs.
- Training history is visualized using matplotlib.

---

## 📝 Notes

- `np.exp()` is used instead of `math.exp()` for better array performance.
- Z-score threshold is set to `3` to remove extreme outliers.
- Bias can be included or excluded by editing one line of code.
- Weights are randomly initialized and printed after training.
- You can export normalized values by uncommenting the `df.to_csv()` section.

---

## ⚙️ Customization

You can easily change these hyperparameters in the code:

| Parameter        | Description                      | Variable Name     |
|------------------|----------------------------------|-------------------|
| Learning Rate    | How fast weights update          | `eta`             |
| Momentum         | Helps smooth out updates         | `mom`             |
| Hidden Neurons   | Number of neurons in hidden layer| `n_hidden_neurons`|
| Epochs           | Training duration                | User input        |
| Z-score Threshold| Outlier detection sensitivity    | `threshold`       |

For hyperparameter tuning, consider using `numpy.linspace()` and a loop to automate testing combinations.

---

## 💾 Optional: Save Weights

To re-use the trained model without retraining, consider saving weights:

```python
np.savetxt("hidden_weights.csv", wsh, delimiter=",")
np.savetxt("output_weights.csv", wsy, delimiter=",")
```

---

## ✅ Planned Enhancements

- [ ] Automate hyperparameter tuning
- [ ] Add saving/loading of weights
- [ ] GUI to let user upload CSV and view results
- [ ] Integrate with MATLAB for post-processing visualization

---

## 📌 Author Notes

- Designed for educational and experimental purposes
- Written with debuggability in mind (separate columns and structures)
- Includes Arabic code comments for clarity
- Follows basic MLP backpropagation principles without external ML libraries

---

## 📈 Sample Output

- RMSE printed at each epoch
- Plot of training vs. validation RMSE
- Early stopping trigger message if applicable
- Final weights of the model

---
