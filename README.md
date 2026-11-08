# 2HDM + n Real Singlets Vacuum Stability Framework

Automated framework implemented in **SageMath** to analyze the asymptotic vacuum stability (bounded-from-below) of Two-Higgs-Doublet Models extended with real scalar singlets.

## Prerequisites & Installation
Before running the code, ensure your environment satisfies:
* **SageMath** (v9.5 or higher recommended)
* **Python 3.10+**

Clone the repository and run the scripts directly from your terminal.

## Usage
Execute the interactive command-line interface inside your SageMath terminal:
```bash
sage stability_cli.py
```

The program will interactively:
1. Display environment prerequisites.
2. Prompt you for the number of real scalar singlets ($n \ge 1$).
3. Construct and display the symbolic potential.
4. Prompt you for numerical parameters.
5. Solve stability equations across topological domains and output whether stability was resolved via global $J_4$ evaluation or required local analysis.
6. Present the stationarity equations to classify the different vacua.
