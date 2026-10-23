# Customer churn prediction

> Work in progress. Started October 2026.

Predicting which customers of a telecom company are likely to cancel, using the public IBM Telco Customer Churn dataset (7,043 customers, 21 columns, about 26% churned).

The goal isn't only a good score. It's a model a retention team could act on: the main metric weighs a missed churner as more costly than a wasted discount offer, and the decision threshold comes from that cost, not the default 0.5.

## Plan

- [ ] Download script and data dictionary
- [ ] EDA: churn by contract type, tenure, charges, services
- [ ] Preprocessing pipeline (`ColumnTransformer`) with tests
- [ ] Baseline and logistic regression
- [ ] Decision tree with cross-validated depth
- [ ] Model comparison (ROC/PR curves, confusion matrices)
- [ ] Cost-based threshold, and a look at the customers each model gets wrong
- [ ] CLI and CI

## Stack

Python 3.12, pandas, scikit-learn, matplotlib, pytest, ruff, uv.

## Quick start

```bash
uv sync
make lint
make test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
