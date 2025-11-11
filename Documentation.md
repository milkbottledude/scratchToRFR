# scratchToRFR
I've been using Random Forest Regressor and Decision Tree Regressor models from sklearn and PyTorch for the past year, and they have helped me to create things I can be proud to be a part of, such as the RFR model in [Johorscrape](https://github.com/milkbottledude/woodlands-jb_tracker) or the predictor models I submitted for Kaggle contests. However, despite all this time we have spent together, I haven't really gotten to know them very well.

Hence, I feel the need to understand them on a deeper level. So in this new project I'm embarking on, we'll be creating a Random Forest Regressor from scratch. 

To turn things up a notch, I won't be using trusty ol' Python. Instead, I'll be opting for Javascript, which I just started learning a week ago. This will help me get some JS practice outside of Leetcode, as well as help me become less reliant on Python.

Lets lay out the roadmap:

### Chapter 1: Data
- 1.1: [Bootstrapping rows from scratch](#11-bootstrapping-rows-from-scratch)
- 1.2: [Feature Selection]

### Chapter 2: Tree Model
- 2.1: [Stop Criteria](21-stop-criteria)
- 2.2: [Split Criteria](22-split-criteria)
- 2.3: [Optimizing Split](23-optimizing-split) (Coming Soon!)
- 2.4: [Branch Recursion](24-branch-recurse)
- 2.5: [Tree Prediction](25-tree-prediction)

### Chapter 3: Forest Management
- 3.1: [n_estimators = n trees](31-n_estimators--n-trees)
- 3.2: [Forest Prediction]

### Chapter 4: [Conclusion](#conclusion)


## 📚 Documentation

## Chapter 1 - Data
Random Forest Regressor models bootstrap their data samples before passing them on to their trees. What this means is that only a random percentage of the entire dataset is passed to each tree. Since bootstrap does not exclude rows after they are chosen, duplicate rows can appear within the subset of data as well.

Not to worry though, this is an intended effect and does not induce overfitting. The whole point is to train each tree to be different from each other. Without repeated rows, each tree is getting the same data, just reshuffled and with a small fraction removed.

To do that, we randomly sample from the training dataset with replacement, until the sample dataset has as many rows as the training dataset. This will go to the first tree. Then we do this again for the subsequent trees.

```
js code here
```

First, we get the length of the training dataset, defined here as len_trng_data. Then, we get a random index between 0 and the length to pick out a row from the training dataset and add it to the tree's dataset. We do this until len_sample_data === len_trng_data.
