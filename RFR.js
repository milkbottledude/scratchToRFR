const fs = require("fs");

const calcAvg = (arr) => {
    let sum = 0
    for (const num of arr) {
        sum += Number(num)
    }
    let avg = sum/arr.length
    return avg
} 

// saving dict data
let filePath = 'rfrData_1.json'

// Node Class
class Node {
    constructor(input_rows, features, allCols) {
        this.input_rows = input_rows
        this.ftError = {}
        this.ftThres = {}
        this.colArr = features
        this.allCols = allCols
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
                let ting = (Number(y)-means[i])**2
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
        let pure_rows = true
        for (const ft in this.ftError) {
            if (this.ftError[ft] < smol) {
                smol = this.ftError[ft]
                ind = ft
            } 
        }
        this.bestFt = [ind, this.ftThres[ind]]
        if (smol > 0) {
            pure_rows = false
        }
        return pure_rows
    }
    testThres = (croppedData, binary=false) => {
        let lowest = Infinity
        let thres_val = undefined
        let x_vals = croppedData.map(row => row[0])
        let unique_x = new Set(x_vals)
        unique_x = [...unique_x]
        let unique_thres = new Set()
        for (let i = 1; i < unique_x.length; i++) {
            let x = unique_x[i-1]
            let y = unique_x[i]
            let thres = (x+y)/2
            unique_thres.add(thres)
        }
        // MAKE CHANGE HERE, UNIQ_THRES N ADD ALL IN BTW VALUES FRM UNIQUE_X
        // done
        if (binary === false) {
            for (const i of unique_thres) {
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
            croppedData.sort((a, b) => a[0] - b[0])
            let y_vals = croppedData.map(row => row[1])
            let left = y_vals.slice(0, falses.length)
            let right = y_vals.slice(falses.length)
            lowest = this.calcVar(left, right)           
        }
        return [lowest, thres_val]
    }
    // this first
    loopFts = (colArr) => {
        // console.log(this.input_rows.slice(0, 4))
        for (const ft in this.ftError) {
            let ftInd = colArr.indexOf(ft)
            console.log(this.input_rows)
            let cropData = this.input_rows.map(rows => [rows[ftInd], rows[rows.length-1]])
            let binary = false
            if (typeof cropData[0][0] === 'boolean') {
                binary = true
            }
            let resArr = this.testThres(cropData, binary)
            // console.log(resArr)
            this.ftError[ft] = resArr[0]
            this.ftThres[ft] = resArr[1]
        }
    }
    passOn = () => {
        let chosenFeat = this.bestFt[0]
        let chosenFeatInd = this.allCols.indexOf(chosenFeat)
        let leftData = undefined
        let rightData = undefined
        let threshold = this.bestFt[1]
        if (threshold !== undefined) {
            leftData = this.input_rows.filter(x => x[chosenFeatInd] <= threshold)
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
        return [leftData, rightData, chosenFeatInd, threshold]
    }
}

// Tree Class
class Tree {
    constructor(all_rows, colArr, min_samp_leaf=1) {
        this.btstr_rows = all_rows
        this.min_samp_leaf = min_samp_leaf
        this.no = 0
        this.colArr = colArr
        // this.nodes = new Map()
        this.JSONdata = {}
        // this.leaves = [] // remove in production 
    }

    feature_bagging = (allCols=this.colArr, n_fts=undefined) => {
        let col_names = allCols.slice(0, -1)
        let node_ft_no = n_fts
        if (!node_ft_no) {
            node_ft_no = Math.floor(Math.sqrt(col_names.length))
        }
        let col_names_node = col_names.slice()
        const node_cols = []
        for (let i = 0; i < node_ft_no; i++) {
            let ind = Math.floor(Math.random() * (col_names_node.length))
            let chosen = col_names_node.splice(ind, 1)
            node_cols.push(chosen[0])
        }
        return node_cols
    }

    recur_node = (node_rows=this.btstr_rows, cur=this.JSONdata) => {
        const node = new Node(node_rows, this.feature_bagging(), this.colArr)
        this.no++
        node.loopFts(this.colArr)
        node.pickBest()
        let [left, right, featInd, thres_val] = node.passOn()
        if (Object.keys(cur).length === 0) {
            cur[0] = {'ftInd': featInd, 'threshold': thres_val, 'kids': {}}
            cur = cur[0]['kids']
        } else {
            cur[1] = {'ftInd': featInd, 'threshold': thres_val, 'kids': {}}
            cur = cur[1]['kids']
        }
        // checking if left and right are already pure
        let uniq_left =  new Set()
        let leftY = []

        for (const row of left) {
            uniq_left.add(row[row.length - 1])
            leftY.push(row[row.length - 1])
        }
        let uniq_right = new Set()
        let rightY = []
        for (const row of right) {
            uniq_right.add(row[row.length - 1])
            rightY.push(row[row.length - 1])
        }
        if (uniq_left.size > 1 && leftY.length > this.min_samp_leaf) {
            console.log('commencing left child')
            this.recur_node(left, cur)
        } else {
            console.log('stop criteria met, ending node')
            cur[0] = {'leaf_avg': calcAvg(leftY)}
            // this.leaves.push(calcAvg(leftY))
        }
        if (uniq_right.size > 1 && rightY.length > this.min_samp_leaf) {
            console.log('commencing right child')
            this.recur_node(right, cur)
        } else {
            console.log('stop criteria met, ending node')
            cur[1] = {'leaf_avg': calcAvg(rightY)}
            // this.leaves.push(calcAvg(rightY))
        }
    }


}

// Forest Class
class trainForest {
    constructor(n_estimators, all_rows, min_samp_leaf=1) {
        this.n_estimators = n_estimators
        this.col_row = all_rows.shift()
        // converting number strings to numbers n binary strings to js binary
        this.trngRows = all_rows.map(line =>
            line.map(v => {
                if (v === "True") return true
                if (v === "False") return false
                return Number(v);
            })
        )
        this.JSONdata = []
        this.min_samp_leaf = min_samp_leaf
    }

    bootstrap_rows = (trngRows) => {
        let tree_data = []
        for (let i = 0; i < trngRows.length; i ++) {
            let ran_int = Math.floor(Math.random() * trngRows.length)
            tree_data.push(trngRows[ran_int])
        }
        return tree_data
    }
    trainTrees = () => {
        for (let i = 1; i <= this.n_estimators; i++) {
            let treeInst = new Tree(this.bootstrap_rows(this.trngRows), this.col_row)
            treeInst.recur_node()
            this.JSONdata.push(treeInst.JSONdata)
        }
    }
    toJSON = (dictData=this.JSONdata, filepath=filePath) => {
        fs.writeFileSync(filepath, JSON.stringify(dictData, null, 2));
        console.log("dict to JSON complete");
    }
}

class predForest {
    constructor(unseenRows) {
        this.unseenRows = unseenRows
    }
    fromJSON = (filepath) => {
        const raw = fs.readFileSync(filepath);
        const dict = JSON.parse(raw);
        return dict
    }
    predictRow = (testRow, dataDict, num=0) => {
        let subDict = dataDict[num]
        if ('leaf_avg' in subDict) {
            if (subDict['leaf_avg'] === null) {
                if (num == 0) {
                    return this.predictRow(testRow, dataDict, 1)
                } else {
                    return this.predictRow(testRow, dataDict, 0)
                }
            } else {
                return subDict['leaf_avg']
            }
        } else {
            let ftInd = subDict['ftInd']
            let threshold = subDict['threshold']
            if (testRow[ftInd] <= threshold) {
                return this.predictRow(testRow, subDict['kids'], 0)
            } else {
                return this.predictRow(testRow, subDict['kids'], 1)
            }
        }
    }
    predAll = (unseenRows=this.unseenRows, JSONpath=filePath) => {
        let avgPreds = []
        let dictS = this.fromJSON(JSONpath)
        for (const row of unseenRows) {
            let preds = []
            for (const dict of dictS) {
                let predY = this.predictRow(row, dict)
                preds.push(predY)
            }
            avgPreds.push(calcAvg(preds))            
        }
        console.log(avgPreds)
        return avgPreds
    }
}

// Execution
let training = (trngRows, trng_csv_path="smol_pt2.csv") => {
    // Read CSV as text
    const unclean_csv = fs.readFileSync(trng_csv_path, "utf8");
    const csv = unclean_csv.replace(/\r/g, "");
    let rows = csv.split("\n")
    rows = rows.map(line => line.split(','))

    testForest = new trainForest(3, trngRows)
    testForest.trainTrees()
    testForest.toJSON()
}

// training(rows)

let unseenRow = [8, 318, 135, 3830, 15.2, 79] // for training u pass the y_column too, but for prediction u remove it
let actualY = 18.2
let testPred = new predForest([unseenRow])
let predY = testPred.predAll()
console.log(actualY, predY[0])

