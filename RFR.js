const fs = require("fs");

// Read CSV as text
const unclean_csv = fs.readFileSync("smol_test_data.csv", "utf8");
const csv = unclean_csv.replace(/\r/g, "");

// remove either the last or 2nd last col depending on whether u want to train for jb or wdlands. 
// If the csv is another dataset, den issok
let rows = csv.split("\n")
rows = rows.map(line => line.split(','))

const colArr = rows.shift()
const len_trng_data = rows.length
const bootstrap_rows = (trngRows=rows) => {
    let tree_data = []
    for (let i = 0; i < len_trng_data; i ++) {
        let ran_int = Math.floor(Math.random() * len_trng_data)
        tree_data.push(trngRows[ran_int])
    }
    // console.log(tree_data.length, len_trng_data) // will remove during production
    // console.log(tree_data[0]) // will remove during production
    return tree_data
}

const used_goods = []
// supply for each node
// after removing the y_columns from 'colArr'
const feature_bagging = (col_names=colArr) => {
    const node_ft_no = Math.floor(Math.sqrt(col_names.length))
    let col_names_node = col_names.slice()
    const node_cols = []
    for (let i = 0; i < node_ft_no; i++) {
        ind = Math.floor(Math.random() * (col_names_node.length-1))
        chosen = col_names_node.splice(ind, 1)
        node_cols.push(chosen)
    }
    return node_cols
}

const calcAvg = (arr) => {
    let sum = 0
    for (const num of arr) {
        sum += Number(num)
    }
    let avg = sum/arr.length
    return avg
} 


// Node Object
class Node {
    constructor(input_rows, features) {
        this.input_rows = input_rows
        this.ftError = {}
        this.ftThres = {}
        for (const feat of features) {
            this.ftError[feat] = Infinity
            this.ftThres[feat] = undefined
        }
        this.bestFt = undefined
    }

    calcVar = (grp1, grp2) => {
        let grps = [grp1, grp2]
        let means = [calcAvg(grp1), calcAvg(grp2)]
        let vars = [0, 0]

        for (let i = 0; i < grps.length; i++) {
            let grp = grps[i]
            for (const y of grp) {
                let ting = (y-means[i])**2
                vars[i] += ting
            }
        }
        let div = grp1.length + grp2.length 
        // console.log(vars)
        // console.log(div)
        return (vars[0] + vars[1])/div
    }
    pickBest = () => {
        let smol = Infinity
        let ind = undefined
        for (const ft in this.ftError) {
            if (this.ftError[ft] < smol) {
                smol = this.ftError[ft]
                ind = ft
            } 
        }
        this.bestFt = [ind, this.ftThres[ind]]
    }
    JSONsave = () => {
        // tbc when we need to save the tree's data (threshold values, chosen feats, etc)
    }
    testThres = (croppedData, binary=false) => {
        croppedData.sort((a, b) => a[0] - b[0])
        console.log(croppedData.slice(0, 5))
        let lowest = Infinity
        let thres_val = undefined
        let x_vals = croppedData.map(row => row[0])
        let unique_x = new Set(x_vals)
        if (binary === false) {
            for (const i of unique_x) {
                    let left = croppedData.filter(row => Number(row[0]) <= Number(i))
                    // console.log(left.length)
                    // console.log(left)
                    let right = croppedData.filter(row => Number(row[0]) > Number(i))
                    left = left.map(row => row[1])
                    right = right.map(row => row[1])
                    // console.log(right.length)
                    // console.log(right)
                    let pot = this.calcVar(left, right)
                    // console.log(pot)
                    if (pot < lowest) {
                        lowest = pot
                        thres_val = i
                    }
            }
        } else {
            let falses = x_vals.filter(x => x === false)
            let left = y_vals.slice(0, falses.length)
            let right = y_vals.slice(falses.length)
            lowest = this.calcVar(left, right)           
        }
        return [lowest, thres_val]
    }
    // this first
    loopFts = () => {
        // console.log(this.input_rows.slice(0, 4))
        for (const ft in this.ftError) {
            console.log(ft)
            let ftInd = colArr.indexOf(ft)
            let cropData = this.input_rows.map(rows => [rows[ftInd], rows[rows.length-1]])
            let binary = false
            console.log(cropData.slice(0, 4))
            if (cropData[0][0] === 'True' || cropData[0][0] === 'False') {
                binary = true
            }
            let resArr = this.testThres(cropData, binary)
            // console.log(resArr)
            this.ftError[ft] = resArr[0]
            this.ftThres[ft] = resArr[1]
        }
    }
    passOn = (leftNode, rightNode) => {
        let chosenFeat = this.bestFt[0]
        let chosenFeatInd = colArr.indexOf(chosenFeat)
        let leftData = undefined
        let rightData = undefined
        let threshold = this.bestFt[1]
        if (threshold) {
            leftData = this.input_rows.filter(x => x[chosenFeatInd] < threshold)
            rightData = this.input_rows.filter(x => x[chosenFeatInd] > threshold)
        } else {
            leftData = this.input_rows.filter(x => x[chosenFeatInd] === true)
            rightData = this.input_rows.filter(x => x[chosenFeatInd] === false)
        }
        // output the Node's experiences b4 moving on to new nodes
        // would be boring if we did'nt get to see how the other features fared.
        console.log('Features and their losses for this node. Errors are weighted variances')
        for (const ft in this.ftError) {
            console.log(`Feature: ${ft}, Error: ${this.ftError[ft]}`)
        }
        // new nodes will continue what we started o7
        // const node1 = new Node(leftData, feature_bagging())
        // const node2 = new Node(rightData, feature_bagging())
        return [leftData, rightData]
    }
}

// testing node
let testNode = new Node(bootstrap_rows(), feature_bagging())
// console.log(testNode)
testNode.loopFts()
console.log(testNode)
testNode.pickBest()
console.log(testNode)
console.log(testNode.passOn())

