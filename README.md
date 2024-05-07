# NonstationarySignalAnalysis

[![GitHub](https://img.shields.io/badge/gsantosmdias-NonstationarySignalAnalysis-blue)](https://github.com/gsantosmdias/NonstationarySignalAnalysis)

Nonstationary Signal Analysis is a Python library that provides powerful tools for analyzing non-stationary time series data. This repository offers a collection of functions for instantaneous time-frequency analysis and spectral entropy computation, enabling data scientists and researchers to gain valuable insights into the dynamics of time series signals.

## Introduction

Time series analysis is a fundamental aspect of various fields, including finance, communications, neuroscience, and many others. Understanding the underlying dynamics of non-stationary time series data is crucial for extracting meaningful features and uncovering hidden patterns. One powerful technique for achieving this is through instantaneous time-frequency analysis, which provides valuable insights into the time-varying frequency content of a signal.

Spectral entropy, on the other hand, measures the complexity or irregularity of a signal's frequency spectrum, offering additional information about the distribution of frequencies present in a time series.

## Features

- **Instantaneous Time-Frequency Analysis**
  
  The instantaneous time-frequency analysis functions allow you to capture dynamic frequency changes in a time series. By computing the spectrogram and extracting the instantaneous frequency, you can observe changes in frequency components, identify transients, track modulations, and detect patterns that might not be apparent in the time or frequency domain alone. These functions play a pivotal role in feature engineering, data science tasks, and signal processing applications.

  ![image](https://github.com/gsantosmdias/NonstationarySignalAnalysis/assets/91618118/7b77d49e-d0c4-4c4b-ab61-ff3ec5ee1c64)


- **Spectral Entropy Computation**

  The spectral entropy functions enable you to measure the complexity or irregularity of a signal's frequency spectrum. By quantifying the distribution of frequencies, spectral entropy helps detect hidden patterns, quantify complexity, and enhance feature extraction. It finds applications in noise reduction, feature selection, anomaly detection, and various signal processing tasks.

  ![image](https://github.com/gsantosmdias/NonstationarySignalAnalysis/assets/91618118/2e7428a5-0a62-4121-a24d-1409f8ceea79)


## Notebooks

To showcase the usage and significance of the functions provided by this repository, two notebooks are available:

- **InstantaneousFrequency.ipynb**: This notebook demonstrates the computation of instantaneous frequency and plotting spectrograms. It explains the importance of instantaneous time-frequency analysis, provides practical examples, and highlights the contributions of the relevant functions.

- **SpectralEntropy.ipynb**: This notebook focuses on spectral entropy computation and its relevance in time series analysis. It explores the concept of spectral entropy, presents different approaches to compute it, and showcases its applications in data science.

Please refer to the respective notebooks for detailed explanations and examples.

## Dependencies

- NumPy
- SciPy
- Matplotlib
- Plotly
  
## How to Use

1. Clone the repository:
Please ensure that these dependencies are installed before using the functions.
```
   git clone https://github.com/gsantosmdias/NonstationarySignalAnalysis.git
```
3. Import the Python module:
```python
import signal_analysis
```
## Summary

This repository provides essential tools for nonstationary signal analysis, including functions for instantaneous time-frequency analysis and spectral entropy computation. By leveraging these functions, data scientists and researchers can gain valuable insights into the dynamics of non-stationary time series data, improve feature engineering, and enhance their data science workflows. Explore the notebooks and start analyzing your non-stationary time series.

## Credits 
The calculations from this repository were based in the  [instfreq](https://www.mathworks.com/help/signal/ref/instfreq.html) and [pentropy](https://www.mathworks.com/help/signal/ref/pentropy.html) functions from Matlab.

## Contributing
Contributions to NonstationarySignalAnalysis are welcome! If you find any issues or have suggestions for improvement, please create an issue on GitHub or submit a pull request.
