const samples = [];

for (let x = 0; x <= 5; x += 0.1) {
    samples.push({
        x: [parseFloat(x.toFixed(2))],
        y: [parseFloat((x * x).toFixed(4))]
    });
}

export default {
    name: "square",
    inputSize: 1,
    outputSize: 1,
    task: "regression",
    samples
};
