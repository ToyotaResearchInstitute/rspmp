
# Indirect Shooting Methods for Stochastic Optimal Control using the Rough Stochastic Pontryagin Maximum Principle

## About

Code for reproducing results in
- T. Lew,
"Unified Optimality Conditions for Stochastic Optimal Control in the Rough Path and Itô Frameworks", 2026,
available at [https://arxiv.org/abs/2609.38395](https://arxiv.org/abs/2609.38395).
- T. Lew, "Rough Stochastic Pontryagin Maximum Principle and an Indirect Shooting Method", SIAM Journal on Control and Optimization, 2026,
available at [https://arxiv.org/abs/2502.06726](https://arxiv.org/abs/2502.06726).

![openloop_feedback](/scripts/results/openloop_feedback.png)

## Using this code

To reproduce results, move to the scripts folder ``cd scripts`` and run:

* Example for open-loop optimal control:

```bash
python example_openloop.py
```

* Example for feedback optimal control:

```bash
python example_feedback.py
```

* Compare solutions to open-loop and feedback problems:

```bash
python compare_openloop_feedback.py --solve --plot
```

* Solve the open-loop for different hyperparameters via the direct and indirect methods:

```bash
python compare_hyperparameters.py  --solve-comparison --sample-sizes-sweep
```

* Generate the feedback shooting results:

```bash
python feedback_comparison.py
```

## Setup

This code was tested with Python 3.10.12 on Ubuntu 22.04.5.

We recommend installing the package in a virtual environment. First, run

```bash
python -m venv ./venv
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then, install all dependencies (numpy, scipy, jax, osqp, matplotlib, pytest, tqdm)
by running

```bash
python -m pip install -r requirements.txt
```

and the package can be installed by running

```bash
python -m pip install -e .
```

## Testing

The following command should run successfully:

```bash
python -m pytest
```
