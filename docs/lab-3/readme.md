---
title: Women in Data Science Workshop Lab 3
description: Energy Demand Forecasting with Granite Timeseries (TTM)
logo: images/ibm-blue-background.png
---

# Energy Demand Forecasting with Granite Timeseries (TTM)

TinyTimeMixers (TTMs) are compact pre-trained models for Multivariate Time-Series Forecasting, open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces the notion of the first-ever "tiny" pre-trained models for Time-Series Forecasting. TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting and can easily be fine-tuned for multi-variate forecasts.

### Install the TSFM Library 

The [granite-tsfm library](https://github.com/ibm-granite/granite-tsfm) provides utilities for working with Time Series Foundation Models (TSFM). Here the pinned version is retrieved and installed.

# Install the tsfm library
```python

import sys
if 'google.colab' in sys.modules:
    ! pip install --force-reinstall --no-warn-conflicts "numpy<2.1.0,>=2.0.2"
```

1. This code snippet uses the "granite-tsfm[notebooks]==v0.2.22" library.
2. The code imports the sys module and 
3. Checks if the google.colab module is present in the sys.modules dictionary and if found, it installs the numpy library with a specific version range (<2.1.0, >=2.0.2) using the pip  package manager. The --force-reinstall flag forces a reinstallation of the package, and the --no-warn-conflicts flag suppresses warnings about conflicting dependencies.

### Import Packages
From `tsfm_public`, we use the TinyTimeMixer model, forecasting pipeline, and plotting function.

```python
import matplotlib.pyplot as plt
import pandas as pd
import torch

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)
from tsfm_public.toolkit.visualization import plot_predictions
```

1. The first three lines in the snippet of code imports the matplotlib.pyplot module as plt, the pandas library as pd and the torch library. These libraries are utilizied for the data visualizations, dataframes and working with tensors which provides-multidimensional arrays respectively.
2. The fourth line imports the TimeSeriesForecastingPipeline and TinyTimeMixerForPrediction classes from the tsfm_public module. These classes are part of the tsfm_public library, which is a collection of time series forecasting models and utilities.
3. The fifth line imports the plot_predictions function from the visualization
submodule of the tsfm_public library. 

This function is used to visualize the forecasted time series data.This code snippet imports the necessary libraries and classes for working with time series data, performing forecasting, and visualizing the results. The imported libraries and classes include data manipulation (Pandas), data visualization (Matplotlib), deep learning (Torch), and time series forecasting (tsfm_public).

### Download the data

We'll work with a dataset of hourly electrical demand, generation by type, and weather in Spain. This dataset was originally available from Kaggle. To simplify access to the data, we will make use of the ([energy consumption](https://huggingface.co/datasets/vitaliy-sharandin/energy-consumption-hourly-spain) and [weather](https://huggingface.co/datasets/vitaliy-sharandin/energy-consumption-weather-hourly-spain)) datasets on Hugging Face.

```python
DATA_FILE_PATH = "hf://datasets/vitaliy-sharandin/energy-consumption-hourly-spain/energy_dataset.csv"
```
The code snippet defines a variable named DATA_FILE_PATH and assigns it a string value. This is a Hugging Face dataset identifier, which follows the format hf://datasets/username/dataset_name/dataset_file.format

`hf://datasets/vitaliy-sharandin/energy-consumption-hourly-spain/energy_dataset.csv`:
points to a specific CSV file containing energy consumption data for Spain, with hourly granularity. 


### Specify time and output variables

We provide the names of the timestamp column and the target column to be predicted. The context length (in time steps) is set to match the pretrained model.

```python
timestamp_column = "time"
target_columns = ["total load actual"]
context_length = 512
```

1. `timestamp_column`: This variable is assigned the string value "time". It represents the name of the column in the dataset that contains the timestamp information for each data point. In this case, the timestamp column is named "time".
2. `target_columns`: This variable is assigned a list containing the string value "total load actual". It represents the name of the column(s) in the dataset that contain the target variable(s) for the forecasting task. In this case, the target column is named "total load actual", which likely represents the actual total energy consumption at each time point.
3. `context_length`: This variable is assigned the integer value 512. It represents the number of time steps that the model will consider as context when generating forecasts. In this case, the context length is set to 512, meaning that the model will use the previous 512 time steps to predict the next time step.

In summary, the code snippet defines three variables: timestamp_column, target_columns, and context_length. These variables are used to specify the timestamp column, target column(s), and context length for a time series forecasting task using the TinyTimeMixer model.

### Read in the data

We parse the csv into a pandas dataframe, filling in any null values, and create a single window containing `context_length` time points. We ensure the timestamp column is a datetime.

```python
# Read in the data from the downloaded file.
input_df = pd.read_csv(
    DATA_FILE_PATH,
    parse_dates=[timestamp_column],  # Parse the timestamp values as dates.
)

# Fill NA/NaN values by propagating the last valid value.
input_df = input_df.ffill()

# Only use the last `context_length` rows for prediction.
input_df = input_df.iloc[-context_length:,]

# Show the last few rows of the dataset.
input_df.tail()
```

1. `pd.read_csv()`: This function from the Pandas library is used to read a CSV file into a DataFrame. In this case, it reads the data from the file specified by `DATA_FILE_PATH`.
2. `parse_dates=[timestamp_column]`: This argument is passed to the `pd.read_csv()` function to specify that the column named `timestamp_column` should be parsed as dates. This ensures that the timestamp values are treated as datetime objects, allowing for more accurate time-based operations and visualizations.
3. `input_df.ffill()`: This method is called on the `input_df` DataFrame to fill any missing or NaN values by propagating the last valid value. This is a common technique for handling missing data in time series datasets, as it maintains the temporal order of the data.
4. `input_df.iloc[-context_length:, ]`: This method is used to select the last `context_length` rows from the `input_df` DataFrame. This ensures that only the most recent data is used for prediction, as the TinyTimeMixer model requires a fixed-length context window.
5. `input_df.tail()`: This method is called on the `input_df` DataFrame to display the last few rows of the dataset. This can be useful for quickly verifying that the data has been correctly loaded, preprocessed, and filtered.

In summary, the code snippet reads a CSV file containing time series data, parses the timestamp column as dates, fills missing values using forward propagation, selects the last `context_length` rows for prediction, and displays the last few rows of the dataset. These steps prepare the data for use with the TinyTimeMixer model in a time series forecasting task.


### Plot the target series

Here we inspect a preview of the target time series column.

```python
fig, axs = plt.subplots(len(target_columns), 1, figsize=(10, 2 * len(target_columns)), squeeze=False)
for ax, target_column in zip(axs, target_columns):
    ax[0].plot(input_df[timestamp_column], input_df[target_column])
```

1. `plt.subplots()`: This function from the Matplotlib library is used to create a figure and a set of subplots. In this case, it creates a figure with a grid of subplots, where the number of rows is equal to the length of the `target_columns` list, and the number of columns is 1. The `figsize` argument is used to set the size of the figure, and the `squeeze` argument is set to `False` to ensure that the returned axes object is always a 2D array, even if there is only one subplot.
2. `for ax, target_column in zip(axs, target_columns)`: This loop iterates over the rows of the `axs` array (i.e., the subplots) and the `target_columns` list simultaneously using the `zip()` function. For each iteration, the loop assigns the current subplot (`ax`) and the corresponding target column (`target_column`) to the variables `ax` and `target_column`, respectively.
3. `ax[0].plot(input_df[timestamp_column], input_df[target_column])`: Inside the loop, this line plots the time series data for the current target column on the corresponding subplot. The `input_df[timestamp_column]` expression retrieves the timestamp values, and the `input_df[target_column]` expression retrieves the target column values. The `plot()` method is called on the current subplot (`ax[0]`) to create the line plot.

In summary, the code snippet creates a figure with a grid of subplots, one for each target column, and plots the time series data for each target column on its corresponding subplot. This visualization helps to compare the trends and patterns in the target columns over time.

### Set up zero shot model
The TTM model is hosted on [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1), and is retrieved by the wrapper, `TinyTimeMixerForPrediction`. We have one input channel in this example.

```python
# Instantiate the model.
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on Hugging Face
    num_input_channels=len(target_columns),  # tsp.num_input_channels
)
```
1. `TinyTimeMixerForPrediction.from_pretrained()`: This method is used to instantiate a pre-trained TinyTimeMixer model from the Hugging Face Model Hub. The method takes the following arguments:
	* `"ibm-granite/granite-timeseries-ttm-r2"`: The name of the pre-trained model on the Hugging Face Model Hub. In this case, it is the TinyTimeMixer model pre-trained on time series data by IBM Granite.
	* `num_input_channels`: The number of input channels in the time series data. This value is set to the length of the `target_columns` list, indicating that the model will process multiple target columns as separate input channels.
2. `zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(...)`: This line instantiates the pre-trained TinyTimeMixer model and assigns it to the `zeroshot_model` variable. The model is now ready to be used for forecasting tasks.

In summary, the code snippet instantiates a pre-trained TinyTimeMixer model from the Hugging Face Model Hub, specifying the number of input channels based on the number of target columns in the dataset. The instantiated model is stored in the `zeroshot_model` variable and can be used for generating forecasts.

### Create a forecasting pipeline

Set up the forecasting pipeline with the model, setting `frequency` given our knowledge of the sample frequency. In this example we set `explode_forecasts` to `False`, which keeps each sequence of predictions in a list within the dataframe cells. We then make a forecast on the dataset.

```python

# Create a pipeline.
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    explode_forecasts=False,
    freq="h",
    device=device,  # Specify your local GPU or CPU.
)

# Make a forecast on the target column given the input data.
zeroshot_forecast = pipeline(input_df)
zeroshot_forecast.tail()
```

1. `device = "cuda" if torch.cuda.is_available() else "cpu"`: This line checks if a GPU is available for use and sets the `device` variable accordingly. If a GPU is available, the `device` variable is set to `"cuda"`; otherwise, it is set to `"cpu"`. This ensures that the forecasting pipeline uses the available hardware for faster computation.
2. `TimeSeriesForecastingPipeline`: This is a class from the `tsfm_public` library that represents a high-level interface for building and executing time series forecasting pipelines. The pipeline consists of several components, such as data preprocessing, feature engineering, model training, and forecast generation.
3. `pipeline = TimeSeriesForecastingPipeline(...)`: This line creates an instance of the `TimeSeriesForecastingPipeline` class, passing the required arguments to configure the pipeline. The arguments include:
	* `zeroshot_model`: The TinyTimeMixer model used for forecasting.
	* `timestamp_column`: The name of the column containing the timestamp information.
	* `id_columns`: A list of columns that do not contain time series data and should not be used for forecasting. In this case, it is an empty list.
	* `target_columns`: A list of columns containing the target time series data for forecasting.
	* `explode_forecasts`: A boolean flag indicating whether to generate separate forecasts for each unique identifier in the `id_columns`. In this case, it is set to `False`.
	* `freq`: The frequency of the time series data, specified as a string (e.g., `"h"` for hourly data).
	* `device`: The device to use for computation, either `"cuda"` for a GPU or `"cpu"` for the CPU.
4. `zeroshot_forecast = pipeline(input_df)`: This line uses the configured pipeline to generate forecasts for the target columns in the `input_df` DataFrame. The resulting forecasts are stored in the `zeroshot_forecast` variable.
5. `zeroshot_forecast.tail()`: This method is called on the `zeroshot_forecast` object to display the last few rows of the forecasted data. This can be useful for quickly verifying that the forecasts have been generated correctly.

In summary, the code snippet creates a time series forecasting pipeline using the TinyTimeMixer model, configures the pipeline with the required parameters, generates forecasts for the target columns in the input data, and displays the last few rows of the forecasted data.


### Plot predictions along with the historical data.

The predicted series picks up where the historical data ends, and we can see that it predicts a continuation of the cyclical pattern and an upward trend.

```python
# Plot the historical data and predicted series.
plot_predictions(
    input_df=input_df,
    predictions_df=zeroshot_forecast,
    freq="h",
    timestamp_column=timestamp_column,
    channel=target_column,
    indices=[-1],
    num_plots=1,
)
```

1. `plot_predictions()`: This function from the `tsfm_public.toolkit.visualization` module is used to visualize the historical data and predicted series. It takes several arguments to customize the plot:
	* `input_df`: The input DataFrame containing the historical time series data.
	* `predictions_df`: The DataFrame containing the forecasted series.
	* `freq`: The frequency of the time series data, specified as a string (e.g., `"h"` for hourly data).
	* `timestamp_column`: The name of the column containing the timestamp information.
	* `channel`: The name of the target column for which the forecasts were generated.
	* `indices`: A list of indices specifying which forecasts to plot. In this case, it is set to `[-1]`, which means that only the most recent forecast is plotted.
	* `num_plots`: The number of subplots to create. In this case, it is set to `1`, indicating that a single plot is generated.
2. `plot_predictions(...)`: This line calls the `plot_predictions()` function, passing the required arguments to customize the plot. The function generates a plot that displays the historical data and the most recent forecast for the specified target column.

In summary, the code snippet uses the `plot_predictions()` function to visualize the historical data and the most recent forecast for the target column in a single plot. This visualization helps to compare the actual data with the predicted series and assess the model's performance.