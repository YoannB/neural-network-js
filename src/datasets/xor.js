export default {
    name: "xor",
    inputSize: 2,
    outputSize: 1,
    task: "binary-classification",

    samples: [
        { x: [0, 0], y: [0] },
        { x: [0, 1], y: [1] },
        { x: [1, 0], y: [1] },
        { x: [1, 1], y: [0] }
    ]
};
