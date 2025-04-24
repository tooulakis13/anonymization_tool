# Data anonymization tool

The tool implements three fundamental privacy-preserving techniques: k-anonymity, t-closeness, and differential privacy.

## Three anonymization methods:

- k-anonymity with generalization and suppression
- t-closeness with Earth Mover's Distance (EMD) metric
- Differential privacy using Laplace mechanism

## Key features:

- Interactive GUI for parameter configuration
- Comprehensive metrics calculation
- Visualization capabilities
- Data export functionality
- Progress tracking

## Technical stack:

- Python 3 with PyQt5 for GUI
- pandas for data manipulation
- scikit-learn for utility metrics
- matplotlib/pyqtgraph for visualization

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Run

Run the tool using:

```bash
python .\anonymization_tool.py
