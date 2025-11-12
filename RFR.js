const fs = require("fs");

// Read CSV as text
const unclean_csv = fs.readFileSync("../Project-JBridge/final_data_tillratings7.csv", "utf8");
const csv = unclean_csv.replace(/\r/g, "");


const rows = csv.split("\n");
const col_names = rows.shift().split(',')
const len_trng_data = rows.length
let tree_data = []
for (let i = 0; i < len_trng_data; i ++) {
    let ran_int = Math.floor(Math.random(len_trng_data))
    tree_data.push(rows[ran_int].split(','))
}

console.log(tree_data.length, len_trng_data)
console.log(tree_data[0])

// for each node
// after removing the y_columns from 'col_names'
const node_ft_no = Math.floor(Math.sqrt(col_names.length))
let col_names_node = col_names.slice()
const node_cols = []
for (let i = 0; i < node_ft_no; i++) {
    ind = Math.floor(Math.random(col_names_node.length))
    chosen = col_names_node.splice(ind, 1)
    node_cols.push(chosen)
}

// STOP CRITERIA NEXT, or should we move stop criteria to after we make the tree, cos its kind of small part if u dont count the part where u integrating w tree