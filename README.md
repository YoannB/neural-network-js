# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

L'historique git a été reconstitué à partir des versions successives du projet, à leurs dates d'origine (janvier 2026). Chaque étape a son tag ; le README de chaque tag présente l'étape correspondante, avec des exemples d'exécution réels.

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| **`step-06-images`** | **Classification d'images** ← ce tag |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 6 — Classification d'images

Le framework de l'étape 5 classe maintenant des photos. Le jeu de données `images` (`src/datasets/images.js`) :

- lit les fichiers `.jpg` du dossier `tmp/oranges/` ;
- les réduit en 32 × 32 pixels, en niveaux de gris, avec la bibliothèque `sharp` : chaque image devient un vecteur de 1024 valeurs entre 0 et 1 ;
- prend l'étiquette dans le nom du fichier : `_ok` (classe 1) ou `_ko` (classe 0).

`exec` accepte directement un fichier image, et `table` affiche la précision ainsi que les faux positifs et faux négatifs.

## Préparer les images

```sh
npm install
mkdir -p tmp/oranges
```

Le dossier `tmp/oranges/` est lu au démarrage de `train` et `table` : il doit exister, même vide, pour utiliser les autres jeux de données (`square`, `xor`, `primes`).

Pour le remplir, `scripts/download-oranges.js` télécharge jusqu'à 100 photos d'oranges depuis Pixabay (clé gratuite à créer sur [pixabay.com/api/docs](https://pixabay.com/api/docs/)) :

```sh
PIXABAY_API_KEY=votre_clé node scripts/download-oranges.js
```

Les fichiers téléchargés s'appellent `orange_N.jpg` ; il reste à les étiqueter en les renommant `orange_N_ok.jpg` ou `orange_N_ko.jpg`.

## Lancer

```sh
npm run train -- images
npm run table
npm run exec -- chemin/vers/photo.jpg
```

## Exemples d'exécution

Sur 100 photos d'oranges étiquetées `ok` / `ko` :

```
$ npm run train -- images
Images chargées: 100, ignorées: 0
…
Epoch 9000/10000 - Loss: 0.002556
Epoch 9500/10000 - Loss: 0.002461
Epoch 10000/10000 - Loss: 0.002375
Model saved to model/model.json

$ npm run table          (résumé)
Résumé:
Task           : binary-classification
Samples        : 100
Accuracy       : 100.00 %
False positives: 0
False negatives: 0
```

Puis sur trois photos qui ne font pas partie de l'entraînement :

```
$ npm run exec -- tmp/test-orange.jpg
Model dataset: images
Expected input size: 1024

Activations cachées :
  Neurone 0 | z=-0.993343 | activation=-0.758784
  …
  Neurone 11 | z=0.418847 | activation=0.395959

Sortie finale y ≈ 0.988827

$ npm run exec -- tmp/test-clementine.jpg
…
Sortie finale y ≈ 0.771983

$ npm run exec -- tmp/test-abricot.jpg
…
Sortie finale y ≈ 0.078741
```

La précision de `table` est mesurée sur les images d'entraînement elles-mêmes.
