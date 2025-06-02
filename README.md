# Web Traffic Forecasting - Deep Learning Model for Time Series Data

This repository demonstrates how to build a deep learning model to forecast web traffic using historical data. By predicting the number of sessions in the next hour, website administrators can dynamically manage resources and improve scalability.

## Problem Statement

To efficiently manage website resources, it's essential to predict the expected number of visitors at different time intervals. The goal is to forecast web traffic (number of sessions) for the next hour based on historical hourly data.

## Approach

This project uses a deep learning approach (Conv1D neural network) to forecast the next hour's web traffic by learning patterns from the past week's data (168 hours).

## Project Steps

### 1. Load the Dataset

The data is loaded from a CSV file (`webtraffic.csv`) with columns:
- `Hour Index`: Sequential hour number.
- `Sessions`: Number of sessions in that hour.

### 2. Data Exploration

- View shape and a preview of the data.
- Visualize the entire web traffic time series and zoom in on the first week's traffic for pattern analysis.

### 3. Data Preparation

- The model predicts the next hour’s traffic based on the previous 168 hours.
- Prepare input-output pairs: inputs are sequences of 168 values, outputs are the next value.
- Split data into training and validation sets (90%/10%).

### 4. Normalization

- Inputs and outputs are normalized using `StandardScaler` to speed up training and improve convergence.

### 5. Model Architecture

A sequential deep learning model built with Keras:
- **Conv1D Layer**: 64 filters, kernel size 3, ReLU activation.
- **Conv1D Layer**: 32 filters, kernel size 5, ReLU activation.
- **Flatten Layer**
- **Dense Layer**: 64 units, ReLU activation.
- **Output Layer**: 1 unit, linear activation.

### 6. Training

- Compile the model with Mean Squared Error (MSE) loss and Adam optimizer.
- Use model checkpointing to save the best model.
- Train for 30 epochs with batch size 32.

### 7. Evaluation & Forecast

- The training and validation loss are monitored to ensure good generalization.
- The best model (with the lowest validation loss) is saved for inference.

## Usage

1. Clone the repository and install requirements.
2. Place your historical web traffic data in `webtraffic.csv` format.
3. Run the notebook `Forecasting_Web_Traffic_1.ipynb` to train and evaluate the model.

## Example Data

| Hour Index | Sessions    |
|------------|-------------|
| 0          | 1418159421  |
| 1          | 1113769116  |
| ...        | ...         |

## Requirements

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- keras / tensorflow

## Reference

Notebook: [`Forecasting_Web_Traffic_1.ipynb`](Forecasting_Web_Traffic_1.ipynb)

---

**Author:** [manola1109](https://github.com/manola1109)  
**License:** MIT
