# Reliable LIME under Query Budget Constraints

## Work Plan

- [x] 1.  **Reproduce(_done_):** This project is going to re implement the core pipeline of `lime_image`, which is the flagship demo in [original paper](https://arxiv.org/pdf/1602.04938).
- [x] 2. **Limitation(_done_):** We will show that Image LIME becomes less reliable under limited query budgets by budget sweep experiments.
- [ ] 3. **Improvement(_in progress_):**
- Budgets Setting: `budgets = [200, 500, 1000, 2000, 3000, 5000]`
- Check resuslts in section `3.2 Coarse-to-Fine Budget Allocation`. Compared with vanilla, the most primary metric `faith_drop_k_mean` is largely improved! However, there're two metrics get worse.
     - `stability_jaccard`: It's worse but still increases generally as budget growing.
     - `fidelity_wmse_mean`: It's very much worse, and it even increases as budget gowing!  
     Possible reasons: `fidelity_wmse_mean` may be unfair for Coarse-to-fine Budget Allocation. Or, the two stages fitting makes the training pairs of explainer largely unlinear.

     Please select a metric that you want to study, find why it becomes worse, and try to fix it if it's possible.
     You can find definitions of these metrics in section `2.2 Numerical metrics`.  
     

     | Name | Current Task |
     | ------------ | ------------ |
     | Qihang Feng | Prepare presentation and report |
     | Wonki Byun |  |
     | Zixi Chai |  |  

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
