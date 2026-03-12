# Reliable LIME under Query Budget Constraints

## What we have done

- [x] 1.  **Reproduce(_done_):** This project re implemented the core pipeline of `lime_image`, which is the flagship demo in [original paper](https://arxiv.org/pdf/1602.04938).
- [x] 2. **Limitation(_done_):** We showed that Image LIME becomes less reliable under limited query budgets by budget sweep experiments with numerical metrics.
- [x] 3. **Improvement(_done_):** We defined four metrics, built budegt sweep experiment pipeline and impelement the coarse to fine budget allocation strategy.

## Current work plan

Now, we are preparing the presentation and the final report.
| Task | DDL |
| ------------ | ------------ |
| Presentation | April 2 |  
| Report | April 27 |

| Name | Current Task |
| ------------ | ------------ |
| Qihang Feng | 1, 3 |
| Wonki Byun | 2, `fidelity_wmse_mean` discussion in 4.3 |
| Zixi Chai | 4, 5 |

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
