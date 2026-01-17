import fs from "fs/promises";
import path from "path";

const API_KEY = process.env.PIXABAY_API_KEY;
if (!API_KEY) {
    console.error("Définis PIXABAY_API_KEY dans les variables d'environnement");
    process.exit(1);
}

const SEARCH_QUERY = "orange fruit";
const PER_PAGE = 100; // récupère directement 100 images
const DIR = "./tmp/oranges";

async function main() {
    // Crée le dossier si nécessaire
    try {
        await fs.mkdir(DIR, { recursive: true });
    } catch { }

    // Appel API Pixabay
    const url = new URL("https://pixabay.com/api/");
    url.searchParams.set("key", API_KEY);
    url.searchParams.set("q", SEARCH_QUERY);
    url.searchParams.set("image_type", "photo");
    url.searchParams.set("per_page", PER_PAGE.toString());

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`Pixabay API erreur: ${res.status}`);
    const data = await res.json();
    const images = data.hits.slice(0, PER_PAGE);

    console.log(`Téléchargement de ${images.length} images...`);

    // Téléchargement séquentiel
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const imgUrl = img.largeImageURL || img.webformatURL;
        const filepath = path.join(DIR, `orange_${i + 1}.jpg`);

        try {
            const imgRes = await fetch(imgUrl);
            if (!imgRes.ok) throw new Error(`Erreur téléchargement: ${imgRes.status}`);
            const buffer = Buffer.from(await imgRes.arrayBuffer());
            await fs.writeFile(filepath, buffer);
            console.log(`✔ ${filepath}`);
        } catch (err) {
            console.error(`✖ Erreur image ${i + 1}: ${err.message}`);
        }
    }

    console.log("Terminé.");
}

main().catch(err => console.error(err));
