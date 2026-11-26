import fs from "fs/promises";
import { existsSync, readdirSync } from "fs";
import path from "path";

// La clé peut venir de .env.local ou .env (Node 20.12 ou plus), ou du terminal.
for (const file of [".env.local", ".env"]) {
    if (existsSync(file) && typeof process.loadEnvFile === "function") {
        process.loadEnvFile(file);
    }
}

const API_KEY = process.env.PIXABAY_API_KEY ?? "";
if (!API_KEY) {
    console.error("Définis PIXABAY_API_KEY dans .env.local ou dans les variables d'environnement");
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

    // Une image déjà présente (triée ou non) n'est pas téléchargée de nouveau.
    const existing = new Set(
        readdirSync(DIR).map(f => f.match(/^orange_(\d+)[_.]/)?.[1]).filter(Boolean)
    );

    console.log(`Téléchargement de ${images.length} images...`);

    // Téléchargement séquentiel
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const imgUrl = img.largeImageURL || img.webformatURL;
        if (existing.has(String(i + 1))) {
            console.log(`= orange_${i + 1} déjà présente`);
            continue;
        }

        // Nommée _ko par défaut : renommer en _ok les images qui montrent une orange.
        const filepath = path.join(DIR, `orange_${i + 1}_ko.jpg`);

        try {
            const imgRes = await fetch(imgUrl);
            if (!imgRes.ok) throw new Error(`Erreur téléchargement: ${imgRes.status}`);
            const buffer = Buffer.from(await imgRes.arrayBuffer());
            await fs.writeFile(filepath, buffer);
            console.log(`✔ ${filepath}`);
        } catch (err) {
            console.error(`✖ Erreur image ${i + 1}: ${(err as Error).message}`);
        }
    }

    console.log("Terminé. Renomme en orange_N_ok.jpg les images qui montrent une orange.");
}

main().catch(err => console.error(err));
