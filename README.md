# Reliable LIME under Query Budget Constraints

## Work Plan

- [x] 1.  **Reproduce(_done_):** This project is going to re implement the core pipeline of `lime_image`, which is the flagship demo in [original paper](https://arxiv.org/pdf/1602.04938).
- [x] 2. **Limitation(_done_):** We will show that Image LIME becomes less reliable under limited query budgets by budget sweep experiments.
- [ ] 3. **Improvement(_in progress_):**
- Please check **section 3.1** in `lime_budget_constraints.ipynb`, and pick one improvement you're interested in.
- Recommended options are `2.2 Repeated LIME with Split Budget and Aggregation` and `3. Balanced Mask Sampling`. 
- **_Please also check section 2.3 to understand how to use `budget_sweep_metrics()` function._** You need to do Budget Sweep Metrics Experiemtns to measure the performance of your improvement. And save your resuslts in `./results` file.
- Budgets Setting: `budgets = [200, 500, 1000, 2000, 3000, 5000]`
     | Name | Current Task |
     | ------------ | ------------ |
     | Qihang Feng | Prepare presentation and report |
     | Wonki Byun | Improvement x.x |
     | Zixi Chai | Improvement x.x |  

     Advice from TA: It's ok if the improvement does't improve the performance, and we can try to find why it does't work.

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
