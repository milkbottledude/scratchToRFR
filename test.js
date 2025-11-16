const calcAvg = (arr) => {
    let sum = 0
    for (const num of arr) {
        sum += Number(num)
    }
    let avg = sum/arr.length
    return avg
} 

const calcVar = (grp1, grp2) => {
    let grps = [grp1, grp2]
    let means = [calcAvg(grp1), calcAvg(grp2)]
    let vars = [0, 0]

    for (let i = 0; i < grps.length; i++) {
        let grp = grps[i]
        for (const y of grp) {
            console.log(y)
            let ting = (y-means[i])**2
            vars[i] += ting
        }
    }
    let div = grp1.length + grp2.length 
    return (vars[0] + vars[1])/div
}

let pt1 = [
  '1750000',   '1750000',   '400000000',
  '1750000',   '3600000',   '12000',
  '17000',     '8000000',   '33000',
  '10000000',  '12000',     '60000',
  '8000000',   '4500000',   '19000000',
  '88500000',  '4500000',   '200000',
  '19000000',  '7000000',   '35000000',
  '4500000',   '19000000',  '175000',
  '6800000',   '230000000', '4500000',
  '175000',    '4500000',   '4500000',
  '230000000', '175000',    '43500000',
  '40000000',  '40000000',  '100000000',
  '13500000',  '13500000',  '110000',
  '45000',     '20000000'
]

let pt2 = [
  '45000',    '600000',
  '265000',   '50000000',
  '265000',   '55000000',
  '34500000', '900000',
  '900000'
]

console.log(calcVar(pt1, pt2))
// console.log(calcAvg(pt2))