// Transformation d'une image en vecteur d'entrées pour le réseau.
//   gray : pixels en niveaux de gris, size × size valeurs
//   rgb  : pixels en couleur, size × size × 3 valeurs
//   hist : histogrammes de couleur (teinte, saturation, luminosité), 20 valeurs,
//          quelle que soit la taille : la position des couleurs dans l'image est
//          perdue, seule leur répartition compte.
import sharp from "sharp";

export const IMAGE_MODES = ["gray", "rgb", "hist"];

const HUE_BINS = 12;
const SAT_BINS = 4;
const VAL_BINS = 4;
const HIST_PIXELS = 32;

export function imageInputSize({ size = 32, mode = "gray" } = {}) {
    if (mode === "gray") return size * size;
    if (mode === "rgb") return size * size * 3;
    if (mode === "hist") return HUE_BINS + SAT_BINS + VAL_BINS;
    throw new Error(`Mode d'image inconnu: ${mode} (disponibles : ${IMAGE_MODES.join(", ")})`);
}

// flop : miroir horizontal (pour l'augmentation de données).
export async function imageToVector(file, { size = 32, mode = "gray", flop = false } = {}) {
    imageInputSize({ size, mode }); // vérifie le mode

    const side = mode === "hist" ? HIST_PIXELS : size;
    let image = sharp(file).resize(side, side, { fit: "fill" }).removeAlpha();
    if (flop) image = image.flop();
    if (mode === "gray") image = image.grayscale();

    const buffer = await image.raw().toBuffer();

    if (mode === "hist") return colorHistogram(buffer);
    return Array.from(buffer, v => v / 255);
}

// Conversion RGB -> HSV (h dans [0, 1[, s et v dans [0, 1]).
function rgbToHsv(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d > 0) {
        if (max === r) h = ((g - b) / d + 6) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h /= 6;
    }
    return [h, max === 0 ? 0 : d / max, max];
}

// Part des pixels dans chaque intervalle de teinte (pixels assez colorés
// seulement), de saturation et de luminosité.
function colorHistogram(buffer) {
    const hue = new Array(HUE_BINS).fill(0);
    const sat = new Array(SAT_BINS).fill(0);
    const val = new Array(VAL_BINS).fill(0);
    const pixels = buffer.length / 3;
    const bin = (v, n) => Math.min(n - 1, Math.floor(v * n));

    for (let i = 0; i < buffer.length; i += 3) {
        const [h, s, v] = rgbToHsv(buffer[i] / 255, buffer[i + 1] / 255, buffer[i + 2] / 255);
        if (s >= 0.25 && v >= 0.2) hue[bin(h, HUE_BINS)]++;
        sat[bin(s, SAT_BINS)]++;
        val[bin(v, VAL_BINS)]++;
    }

    return [...hue, ...sat, ...val].map(c => c / pixels);
}
