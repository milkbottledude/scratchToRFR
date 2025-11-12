# scratchToRFR

Welcome! In this new project I'm embarking on, we'll be creating a Random Forest Regressor from scratch using Javascript.

Lets lay out the roadmap, don't worry its not that long:

### Chapter 1: Data
- 1.1: [Bootstrapping rows from scratch](#11-bootstrapping-rows-from-scratch)
- 1.2: [Feature Bagging](#22-feature-bagging)

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

### 1.1: Bootstrapping Rows from scratch

Random Forest Regressor models bootstrap their data samples before passing them on to their trees. What this means is that only a random percentage of the entire dataset is passed to each tree. Since bootstrap does not exclude rows after they are chosen, duplicate rows can appear within the subset of data as well.

Not to worry though, this is an intended effect and does not induce overfitting. The whole point is to train each tree to be different from each other. Without repeated rows, each tree is getting the same data, just reshuffled and with a small fraction removed.

To do that, we randomly sample from the training dataset with replacement, until the sample dataset has as many rows as the training dataset. This will go to the first tree. Then we do this again for the subsequent trees.

First, we have to extract the information from the data csv. Here, I use a sample csv from another project, just to demonstrate what we will be doing to the actual training data csv.

```
const fs = require("fs");

const unclean_csv = fs.readFileSync("../Project-JBridge/final_data_tillratings7.csv", "utf8");
const csv = unclean_csv.replace(/\r/g, "");
```

We need fs, which is used most of the time when working with other files in the local device. We read it, then remove the '\r' text, which is added at the end of all csv lines (before the '\n' newline text).

As you can see, the process of reading csv files is very different compared to python. In python, courtesy of pandas, we can neatly convert the csv into a pandas dataframe a few simple and short commands. 

Unfortunately, pandas is exclusive to python only, and Javascript's pandas alternatives are a pain to download. So we will just be manually reading them as strings, before parsing them in the next part.


```
const rows = csv.split("\n");
const col_names = rows.shift().split(',')
const len_trng_data = rows.length
let tree_data = []
for (let i = 0; i < len_trng_data; i ++) {
    let ran_int = Math.floor(Math.random(len_trng_data))
    tree_data.push(rows[ran_int].split(','))

console.log(tree_data.length, len_trng_data)
console.log(tree_data.slice(0, 2))
}
```
We split the csv by the '\n' newline text to get each line of data. One line represents a row. Before carrying on, we remove the first row, which are the column names.

We define the length of the training dataset as `len_trng_data`. Then, using Math.random(), we get a random integer between 0 and 'len_trng_data' to pick out a row from the training dataset, before adding it to the tree's dataset. 

We do this until tree_data.length === len_trng_data, lets see if they are the same. While we are at it, we can check out the first row of data too.

```
751 751
[
  '0.0',                 '0.0',
  '1.0',                 '0.0',
  '0.0',                 '0.0',
  '1.0',                 '0.2588190451025207',
  '0.9659258262890684',  '11',
  '20',                  '3',
  '-0.7907757369376986', '-0.6121059825476627',
  '325',                 '0.3375228995941133',
  '0.9413173175128472',  '2024-11-20',
  'False',               'False',
  '0.0',                 '0.0'
]
```
Looks good, the lengths are the same and the row has all 22 column values.

This is just for a single tree. When we expand from individual trees into forests after getting the base tree infrastructure right, we will have to use for loops to give a `tree_data` array for every tree.

Thats the data rows sorted, now for the columns

### 1.2: Feature Bagging
Random Forest Regressor models do what is called 'bagging' to the features to prevent overfitting. Basically, every node in a tree only gets sqrt(n) number of features, where 'n' is the total number of features. 

So if we have 20 features, each split point (node) gets sqrt(20) ~= 4 features. This allows each tree to be truly unique from each other, making for a more robust forest.

The features are sampled without replacement within the nodes, but with replacement for a new node. So every feature has an equal chance of being chosen for a node, regardless whether they appeared in a previous node or not.

```
// for each node
// done after removing the y_columns from 'col_names' ofc
const node_ft_no = Math.floor(Math.sqrt(col_names.length))
let col_names_node = col_names.slice()
const node_cols = []
for (let i = 0; i < node_ft_no; i++) {
    ind = Math.floor(Math.random(col_names_node.length))
    chosen = col_names_node.splice(ind, 1)
    node_cols.push(chosen)
}
```

First we obtain sqrt(n) in line 1, then create a copy of the feature array using slice(). This is because we are sampling without replacement for each node, so we need to remove the options once they are picked without touching the original.

Afterwards, we generate a random number within the length of the feature array copy 'ind'. Then, in one line of code, remove that index from the feature array copy and define the removed feature using `splice(ind, 1)` as 'chosen', before adding 'chosen' to the node_cols array.

Again, we will have to use a for loop to do this for every node of every tree, so thats a nested for loop. Perhaps I should have considered the toil this would take on my laptop, especially since we might be running many trees and even more rows of data.

## Chapter 2 - Tree Model

### 2.1: Stop Criteria


