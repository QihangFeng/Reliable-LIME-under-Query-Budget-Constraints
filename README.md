# Reliable LIME under Query Budget Constraints

## Work Plan

 - [x] 1. **Reproduce(*done*):** This project is going to re implement the core pipeline of `lime_image`, which is the flagship demo in [original paper](https://arxiv.org/pdf/1602.04938).
- [x] 2. **Limitation(*done*):** We will show that Image LIME becomes less reliable under limited query budgets by budget sweep experiments.
- [ ] 3. **Improvement(*in progress*):** Please check **3.1 section** in `lime_budget_constraints.ipynb`.


## Environment setup

| Package      | Version      |
| ------------ | ------------ |
| python       | 3.8.10       |
| numpy        | 1.24.4       |
| scipy        | 1.10.1       |
| scikit-learn | 1.0.2        |
| scikit-image | 0.21.0       |
| matplotlib   | 3.7.5        |
| PyTorch      | 2.2.2+cu121  |
| torchvision  | 0.17.2+cu121 |

## Reference

1. Ribeiro M T, Singh S, Guestrin C. " Why should i trust you?" Explaining the predictions of any classifier[C]//Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016: 1135-1144.
2. Official LIME repository: https://github.com/marcotcr/lime
3. Reference implementation used in this project: https://github.com/marcotcr/lime/blob/master/lime/lime_image.py

## Authors

- Qihang Feng qfeng5@ualberta.ca
- Wonki Byun wonki@ualberta.ca
- Zixi Chai zchai3@ualberta.ca
