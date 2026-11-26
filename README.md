# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

## Branche `claude-improvement`

Tu es sur la branche `claude-improvement` : c'est la reprise du projet par Claude, à partir de l'étape 8. Elle fait évoluer le code vers un mini framework testé, écrit en TypeScript et utilisable comme bibliothèque.

- La branche `main` reste le projet de base (étapes 1 à 8) et ne reçoit pas ces évolutions.
- Chaque évolution arrive par une pull request relue avant d'être fusionnée dans cette branche, et chaque étape franchie porte un tag `claude-NN-…`.
- Pour revenir au projet de base : `git switch main`.

## Origine du projet

- **Le code des étapes 1 à 8** a été écrit en janvier 2026 avec l'aide d'une IA (ChatGPT). Une partie de l'étape 7 provient d'une version écrite avec Claude ; le commit de cette étape porte donc un co-auteur supplémentaire.
- **L'historique git** a été reconstitué en novembre 2026 à partir des dossiers de versions datés : chaque étape est un commit à la date de modification de ses fichiers, avec un tag.
- **Les README de chaque tag** ont été rédigés à ce moment-là avec Claude, d'où la ligne `Co-Authored-By` sur les commits. Les exemples d'exécution qu'ils montrent sont de vraies exécutions.
- **Les étapes `claude-NN`** sont écrites par Claude à partir de novembre 2026.

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse ← version de `main` |
| `claude-01-fiabilite` | Corrections, graine aléatoire, séparation entraînement / test, tests |
| `claude-02-api` | Couches cachées configurables, sauvegarde complète des optimiseurs, nettoyage |
| `claude-03-images` | Évaluation sur plusieurs graines, options d'image, augmentation, L2, arrêt sur validation |
| `claude-04-typescript` | Portage en TypeScript strict, comportement identique |
| **`claude-05-framework`** | **Mini framework : API publique, paquet sans dépendance, exemples** |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 22 ou plus pour cette branche (Node.js 18 suffit pour les tags précédents).

---

## Lancer

```sh
npm install
npm test                  # compile puis lance les tests
npm run train -- rps
npm run table -- rps
npm run exec -- rps 1 0 0
```

Exemples d'utilisation de la bibliothèque : `npm run example:xor` et `npm run example:sine` (voir l'étape `claude-05`).

Options de `train` et `evaluate` :

- réseau et entraînement : `--epochs`, `--layers` (par exemple `16,8`), `--neurons` (une seule couche), `--activation`, `--lr`, `--noise`, `--optimizer` (`sgd`, `adam`, `rmsprop`), `--seed`, `--target-loss` ;
- contre l'apprentissage par cœur : `--l2`, `--validation` (part des exemples d'entraînement mise de côté, par exemple `0.2`), `--patience`, `--augment flip` ;
- jeu `images` : `--image-size`, `--image-mode` (`gray`, `rgb`, `hist`).

`npm run evaluate -- <dataset> --seeds 1-10 [options]` entraîne une fois par graine et donne la moyenne et la dispersion du score sur les exemples de test.

Le code est en TypeScript, dans `src/` : chaque script npm le compile d'abord vers `dist/` (`npm run build` le fait seul).

Sous Windows PowerShell 5.1, `npm run train -- rps --epochs 500` peut perdre le `--`, et les options ne sont alors pas transmises. Dans ce cas, compile une fois puis appelle les scripts compilés : `npm run build`, puis `node dist/src/train.js rps --epochs 500`.

Jeux de données : `linear`, `square`, `xor`, `primes`, `rps`, `images`.

### Préparer le jeu `images`

Le jeu `images` lit les fichiers `.jpg` du dossier `tmp/oranges/`, qui n'est pas versionné. L'étiquette vient du nom du fichier : `_ok` dans le nom pour une orange (classe 1), sinon une autre image (classe 0).

1. Crée une clé d'API gratuite sur [Pixabay](https://pixabay.com/api/docs/) et mets-la dans un fichier `.env.local` à la racine du projet. Ce fichier n'est pas versionné : le `.gitignore` exclut `.env` et `.env.*`.

   ```
   PIXABAY_API_KEY=ta_clé
   ```

2. Télécharge 100 images. Le script lit `.env.local` (ou `.env`) lui-même ; tu peux aussi définir `PIXABAY_API_KEY` dans le terminal. Une image déjà présente n'est pas téléchargée de nouveau.

   ```sh
   npm run download-images
   ```

3. Trie les images à la main. Le script les nomme `orange_1_ko.jpg`, `orange_2_ko.jpg`…, c'est-à-dire toutes de la classe 0. Renomme en `_ok` celles qui montrent une orange, par exemple `orange_12_ok.jpg`. Sans ce tri, le réseau n'aurait rien à apprendre.

4. Entraîne puis évalue. Avec la descente de gradient simple, la perte baisse régulièrement ; avec Adam, elle oscille sur ce jeu.

   ```sh
   npm run build
   node dist/src/train.js images --optimizer sgd --epochs 3000
   node dist/src/table.js images
   node dist/src/exec.js images chemin/vers/une-image.jpg
   ```

## claude-05 — Mini framework

Le réseau devient une bibliothèque : un point d'entrée (`src/index.ts`), une API pour entraîner et évaluer, des déclarations de types, et aucune dépendance. Les jeux de données et les scripts `train`, `table`, `exec`, `evaluate` restent dans le dépôt comme outils et exemples.

### Utiliser la bibliothèque

```sh
npm install github:YoannB/neural-network-js#claude-improvement
```

L'installation depuis git compile le paquet (script `prepare`). Node.js 22 ou plus.

```ts
import { Network, type Sample } from "neural-network-js";

const xor: Sample[] = [
    { x: [0, 0], y: [0] }, { x: [0, 1], y: [1] },
    { x: [1, 0], y: [1] }, { x: [1, 1], y: [0] }
];

const net = new Network({ inputSize: 2, hiddenLayers: [8], task: "binary-classification", optimizer: "adam", seed: 1 });
net.fit(xor, { epochs: 10000, targetLoss: 0.001 });

net.predictValue([1, 0]);   // probabilité de la classe 1
net.evaluate(xor);          // { loss, accuracy, samples }

const saved = JSON.stringify(net);          // modèle sauvegardé
const copy = Network.load(JSON.parse(saved));
```

### API

| Élément | Rôle |
|---|---|
| `new Network(options)` | `inputSize`, `hiddenLayers` (par exemple `[16, 8]`), `activation` (`tanh`, `relu`, `leakyrelu`, `sigmoid`), `task` (`regression`, `binary-classification`, `multiclass-classification`), `outputSize`, `optimizer` (`sgd`, `adam`, `rmsprop`), `learningRate`, `l2`, `preprocessing`, `seed` |
| `net.fit(samples, options)` | Entraîne : `epochs`, `targetLoss`, `noise`, `validation` (exemples), `patience`, `evaluateEvery`, `onEpoch`. Renvoie le nombre d'époques, la perte et la raison de l'arrêt. |
| `net.predictValue(x)` | Valeur en régression, probabilité de la classe 1 en classification binaire. |
| `net.predictProbabilities(x)` | Probabilité de chaque classe en multiclasse. |
| `net.evaluate(samples)` | `{ loss, accuracy, samples }` |
| `net.toJSON()`, `Network.load(json)` | Sauvegarde et chargement (les modèles des étapes précédentes se chargent aussi). |
| `splitSamples(samples, { testRatio, seed })` | Séparation entraînement / test avec graine. |
| `computePreprocessing(samples)` | Échelle des entrées, calculée sur les exemples d'entraînement. |
| `createRandom`, `setSeed`, `shuffle` | Générateur aléatoire à graine. |
| `trainNetwork`, `meanLoss`, `accuracy`, `evaluate` | Les mêmes fonctions, hors de la classe. |
| `Layer`, `Neuron`, `activations` | Les briques internes, pour les lire ou les essayer. |

Les types (`Sample`, `Task`, `ModelJSON`…) sont exportés avec.

### Ce qui a changé

- **Point d'entrée** `src/index.ts`, et `Network.fit` / `Network.evaluate` en plus des fonctions existantes.
- **`seed` dans les options du réseau** : l'initialisation des poids utilise son propre générateur, sans toucher au générateur partagé.
- **Cœur sans dépendance** : `src/core` n'importe ni module Node.js ni `sharp` (un test le vérifie), et fonctionne donc aussi dans un navigateur. `sharp` devient une dépendance de développement, utilisée par le jeu `images` et `exec`.
- **Paquet** : `exports` et `types` pointent vers le code compilé ; seul le cœur est publié (36 fichiers, 27 ko). Installé dans un projet vide depuis `npm pack`, il n'apporte aucune dépendance et se compile en TypeScript strict sans les types de Node.js.
- **Exemples** (`examples/`) qui n'utilisent que l'API publique.
- **Tests** : 43, dont les tests de l'API publique. Les 9 entraînements de référence de `claude-04` donnent toujours les mêmes modèles, octet pour octet.

### Exemples (claude-05)

```
$ npm run example:xor          (sans les lignes de npm)
Perte 0.000999 après 2630 époques
0 XOR 0 = 0   prédit : 0.0002
0 XOR 1 = 1   prédit : 0.9989
1 XOR 0 = 1   prédit : 0.9985
1 XOR 1 = 0   prédit : 0.0013
Exactitude : 100 %
```

```
$ npm run example:sine          (sans les lignes de npm)
Époque 500 - perte 0.001791
Époque 1000 - perte 0.000587
Époque 1500 - perte 0.000166
Époque 2000 - perte 0.000151

2000 époques
Écart moyen sur les 160 points d'entraînement : 0.0086
Écart moyen sur les 40 points jamais vus      : 0.0088

x        sin(x)    prédit
 1.721    0.9888    1.0000
-0.837   -0.7424   -0.7346
 1.310    0.9663    0.9826
-0.521   -0.4977   -0.4880
 1.216    0.9376    0.9539
```

Sur `sin(x)`, le réseau est aussi précis sur les points qu'il n'a jamais vus que sur ceux qui ont servi à l'entraînement : il a appris la fonction, pas les points. C'est l'inverse du jeu `images` de l'étape `claude-03`, où le score de test reste proche du hasard.

## claude-04 — TypeScript

Tout le code (réseau, jeux de données, scripts, tests) est porté en TypeScript, en mode strict. Le comportement ne change pas : c'est la condition de cette étape.

- **Types partagés** (`src/core/types.ts`) : `Task`, `Sample`, `Dataset`, `ModelJSON` (le format des modèles sauvegardés, ancien format compris), les activations et les optimiseurs. Une tâche ou une activation mal écrite est refusée à la compilation.
- **`predictValue` et `predictProbabilities`** complètent `predict`, dont le type de retour dépend de la tâche (un nombre, ou un vecteur en multiclasse).
- **Ligne de commande** : un optimiseur, une activation ou une augmentation inconnus donnent une erreur au lieu d'être ignorés.
- **Compilation** : `tsc` (TypeScript 7) compile `src/`, `scripts/` et `test/` vers `dist/`. Les scripts npm compilent avant de s'exécuter. Les tests restent sur `node:test`.
- **Node.js 22 ou plus** : Node 18 et 20 ne sont plus maintenus.

Les sections des étapes précédentes citent les fichiers `.js` de leur tag ; ils sont aujourd'hui en `.ts`, aux mêmes emplacements.

### Vérification : mêmes modèles qu'en JavaScript

Avant le portage, neuf entraînements ont été lancés avec le code JavaScript de `claude-03`, puis relancés avec le code TypeScript compilé, avec les mêmes options :

| Jeu | Options |
|---|---|
| `rps`, `xor` | par défaut |
| `square`, `primes` | `--epochs 2000` |
| `linear` | `--epochs 500 --optimizer sgd` |
| `xor` | `--optimizer rmsprop --layers 6,4 --activation relu --epochs 1000 --l2 0.001` |
| `rps` | `--validation 0.2 --epochs 200 --target-loss 0 --patience 100` |
| `images` | `--image-mode hist --epochs 200` |
| `images` | `--image-size 8 --image-mode rgb --augment flip --epochs 100 --optimizer sgd --validation 0.2` |

Les neuf modèles sauvegardés, les sorties de `table` et celle de `exec` sont identiques octet pour octet ; seules les durées affichées par `train` diffèrent.

### Exemples (claude-04)

```
$ npm test          (extrait)
> neural-network-js@1.0.0 pretest
> tsc

> neural-network-js@1.0.0 test
> node --test "dist/test/*.test.js"

✔ gradients corrects : régression (3.5302ms)
✔ gradients corrects : classification binaire (sigmoïde) (1.0948ms)
✔ gradients corrects : classification multiclasse (softmax) (1.8552ms)
…
✔ reprendre un entraînement adam sauvegardé donne le même modèle (4.1698ms)
✔ reprendre un entraînement rmsprop sauvegardé donne le même modèle (4.4501ms)
…
ℹ tests 37
ℹ pass 37
ℹ fail 0
```

Ce que le compilateur refuse maintenant, sur un fichier d'essai :

```ts
const net = new Network({ inputSize: 2, task: "classification", hiddenLayers: [8] });
console.log(net.predict([1, 0]).toFixed(3));
```

```
error TS2322: Type '"classification"' is not assignable to type 'Task | undefined'.
error TS2339: Property 'toFixed' does not exist on type 'number | number[]'.
  Property 'toFixed' does not exist on type 'number[]'.
```

La tâche s'écrit `"binary-classification"`, et `predict` peut renvoyer un vecteur : `predictValue` renvoie toujours un nombre.

## claude-03 — Images : mesurer avant de conclure

L'étape 1 de cette branche a montré que le réseau apprend par cœur les images d'entraînement sans reconnaître les autres. Cette étape ajoute de quoi mesurer proprement et essaie les pistes habituelles.

- **`evaluate`** (`src/evaluate.js`) : un même réglage est entraîné avec plusieurs graines. Chaque graine change la séparation entraînement / test et l'initialisation, ce qui évite de conclure sur un seul tirage de 20 images de test. L'entraînement commun à `train` et `evaluate` est dans `src/core/experiment.js`.
- **Options d'image** (`src/datasets/image-features.js`) : `--image-size` et `--image-mode` : `gray` (comme avant), `rgb` (la couleur, trois valeurs par pixel) ou `hist` (20 valeurs : la répartition des teintes, saturations et luminosités, sans la position des pixels).
- **Contre l'apprentissage par cœur** : `--augment flip` ajoute les images miroir des exemples d'entraînement ; `--l2` pénalise les grands poids ; `--validation 0.2` met de côté 20 % des exemples d'entraînement, arrête l'entraînement quand leur perte ne s'améliore plus et garde les meilleurs poids.
- **`exec` prépare l'image comme l'entraînement** : il la recadrait alors que le jeu de données l'étire (voir les notes par étape).
- **Tests** : 37 tests, dont les modes d'image sur des images construites, l'effet de L2 et l'arrêt sur validation.

### Résultats sur le jeu `images`

100 images (48 oranges), 80 pour l'entraînement et 20 pour le test, 10 graines, descente de gradient simple, 3000 époques, une couche cachée de 10 neurones :

| Réglage | Train | Test |
|---|---|---|
| `gray` 32 × 32 (comme à l'étape 6) | 100 % | 46,5 % ± 13,0 |
| `gray` 16 × 16 | 100 % | 47,0 % ± 13,5 |
| `rgb` 16 × 16 | 100 % | 56,5 % ± 9,0 |
| `rgb` 16 × 16, `--augment flip` | 100 % | 55,0 % ± 10,7 |
| `rgb` 16 × 16, `--l2 0.001` | 100 % | 56,5 % ± 8,7 |
| `rgb` 16 × 16, `--validation 0.2` | 60,0 % | 53,0 % ± 10,3 |
| `rgb` 16 × 16, les trois | 57,9 % | 49,5 % ± 9,9 |
| `hist` | 80,5 % | 58,5 % ± 10,3 |
| `hist`, Adam | 100 % | 57,5 % ± 11,2 |

Répondre toujours « orange » donnerait environ 52 % sur ces mêmes tirages de test.

**Aucun réglage ne reconnaît vraiment les oranges.** La couleur fait gagner une dizaine de points sur le niveau de gris, mais l'écart entre graines (± 9 à 13 points sur 20 images) reste du même ordre que tous les écarts du tableau. L'augmentation et L2 ne changent rien de mesurable. Avec la validation, l'entraînement s'arrête au bout de 500 époques environ : la perte de validation ne s'améliore presque jamais, signe que ce qui est appris sur les images d'entraînement ne se transfère pas.

La raison tient surtout aux données : les images rejetées (`_ko`) sont elles aussi presque toutes orange (mandarines, abricots, potirons, oranges épluchées ou sanguines). Les distinguer demande de reconnaître une forme et une texture, pas une couleur. Un réseau entièrement connecté, qui voit chaque pixel séparément, n'en est pas capable avec 80 images. Les pistes suivantes sont beaucoup plus d'images et une couche de convolution.

Ces chiffres dépendent des images : avec un autre téléchargement ou un autre tri, ils changeront.

### Exemples (claude-03)

```
$ npm run evaluate -- images --seeds 1-10 --epochs 3000 --optimizer sgd --image-mode hist
Dataset : images
Options : --epochs 3000 --optimizer sgd --image-mode hist

Graine | Époques | Train    | Test
     1 |    3000 |   82.5 % |   50.0 %
     2 |    3000 |   77.5 % |   65.0 %
     3 |    3000 |   73.8 % |   60.0 %
     4 |    3000 |   82.5 % |   55.0 %
     5 |    3000 |   77.5 % |   70.0 %
     6 |    3000 |   87.5 % |   50.0 %
     7 |    3000 |   76.3 % |   80.0 %
     8 |    3000 |   80.0 % |   60.0 %
     9 |    3000 |   83.8 % |   50.0 %
    10 |    3000 |   83.8 % |   45.0 %

Train : 80.5 % ± 4.0
Test  : 58.5 % ± 10.3 (min 45.0, max 80.0)
```

Selon la graine, le score de test va de 45 % à 80 % pour le même réglage : un seul essai ne permet pas de conclure.

## claude-02 — Couches et API

- **Couches cachées configurables** : `--layers 16,8` crée deux couches cachées de 16 puis 8 neurones (`--neurons 10` reste un raccourci pour une seule couche). Dans le code : `new Network({ hiddenLayers: [16, 8], ... })`. La rétropropagation parcourt toutes les couches.
- **Activation cachée au choix** : `--activation` accepte `tanh` (par défaut), `relu`, `leakyrelu` et `sigmoid`.
- **Sauvegarde complète d'Adam et RMSprop** : le modèle enregistre aussi les moments du biais (`mBias`, `vBias`). Reprendre un entraînement depuis un modèle sauvegardé donne exactement le même résultat qu'un entraînement continu.
- **Nettoyage** : `yScale` (calculé mais jamais utilisé), `denormalizeY`, `computeOutputGradient`, `getWeightStats` et `lastError` sont retirés. Les modèles des étapes 5 à 8, au format `hidden` / `output`, se chargent encore.
- **Script d'images** : `scripts/download-oranges.js` lit la clé dans `.env.local` et nomme les images `_ko` par défaut.
- **Tests** : 29 tests, dont la vérification des gradients avec trois couches cachées et pour chaque activation, et la reprise d'un entraînement Adam ou RMSprop.

### Exemples (claude-02)

```
$ npm run train -- rps --layers 8,8 --activation relu          (extrait)
  LAYERS: [ 8, 8 ],
  ACTIVATION: 'relu',
Epoch 1/10000 - Loss: 0.435692
Epoch 9/10000 - Loss: 0.000093
Perte cible 0.0001 atteinte à l'époque 9
Training: 75.764ms
```

```
$ npm run exec -- rps 0 0 1          (extrait)
Hidden layer 2:
Neuron 0 | a=0
Neuron 1 | a=0
Neuron 2 | a=0
Neuron 3 | a=0
Neuron 4 | a=2.704151
Neuron 5 | a=3.577357
Neuron 6 | a=3.839604
Neuron 7 | a=0

Output: [0.999913,0.000075,0.000011]
```

Contre les ciseaux (classe 2), le réseau joue la pierre (classe 0). Avec ReLU, une partie des neurones reste à 0 pour cette entrée.

Plus de couches n'est pas toujours mieux. Sur `square` (x², 51 points, Adam, 10 000 époques, sans arrêt anticipé), une couche de 10 neurones fait mieux que deux couches de 16 et 8 :

| Couches | Graine 1 | Graine 2 | Graine 3 | Graine 42 |
|---|---|---|---|---|
| `10` | 86 % (RMSE 0,051) | 82 % (0,058) | 82 % (0,160) | 88 % (0,077) |
| `16,8` | 24 % (0,958) | 76 % (0,328) | 55 % (0,303) | 29 % (0,576) |

Le pourcentage est celui de `table` : la part des points prédits à 5 % près. La RMSE est l'erreur quadratique moyenne, en unités de y. Avec deux couches, la perte oscille et les neurones de la seconde couche saturent à ±1 : les sorties, jusqu'à 25, ne sont pas normalisées. Normaliser les sorties aide la version à deux couches mais dégrade la version à une couche (RMSE de 0,16 à 0,32 avec Adam) ; ce n'est donc pas fait ici.

## claude-01 — Fiabilité

Cette étape corrige les points relevés dans les [notes par étape](#notes-par-étape) et rend les résultats vérifiables.

- **Graine aléatoire** (`src/core/random.js`) : l'initialisation des poids, le mélange des données et le bruit dépendent de `--seed` (42 par défaut). Deux entraînements avec la même graine donnent le même modèle. La graine est enregistrée dans le modèle.
- **Séparation entraînement / test** (`src/datasets/split.js`) : `rps` et `images` réservent 20 % de leurs exemples au test. `train` n'apprend que sur les 80 % restants et `table` évalue les deux parties séparément.
- **Gradient de la sortie sigmoïde** : en classification binaire, le gradient `p − y` n'est plus multiplié par la dérivée de la sigmoïde. Sur XOR, la perte atteinte en 3000 époques est environ dix fois plus basse qu'avant.
- **Critère d'arrêt** (`src/core/training.js`) : l'entraînement s'arrête quand la perte moyenne descend sous `--target-loss` (0,0001 par défaut, 0 pour désactiver). Les entropies croisées ne peuvent plus être négatives.
- **Jeux de données chargés à la demande** : le dossier `tmp/oranges/` n'est plus nécessaire pour les jeux autres que `images`.
- **Sorties multiclasses** : `table` les lit avec `predict`, et `exec` affiche toutes les probabilités.
- **Tests** : `npm test` lance les tests avec `node:test`, le module de test intégré à Node, sans dépendance supplémentaire. Ils vérifient notamment la rétropropagation par différences finies pour les trois tâches.

### Exemples (claude-01)

```
$ npm test          (extrait)
✔ gradients corrects : régression (4.0939ms)
✔ gradients corrects : classification binaire (sigmoïde) (0.9719ms)
✔ gradients corrects : classification multiclasse (softmax) (1.7226ms)
…
✔ XOR est appris (classification binaire) (66.9726ms)
✔ pierre-feuille-ciseaux : 100 % sur les exemples de test (29.0881ms)
✔ même graine, même modèle (9.9845ms)
…
ℹ tests 21
ℹ pass 21
ℹ fail 0
```

Pierre-feuille-ciseaux, avec la séparation entraînement / test et l'arrêt à la perte cible :

```
$ npm run train -- rps
Dataset    : rps
Model name : rps
Samples    : 600
Input size : 3
Output size: 3
Task       : multiclass-classification
Training config: {
  EPOCHS: 10000,
  NEURONS: 10,
  LR: 0.0025,
  NOISE: 0,
  OPTIMIZER: 'adam',
  SEED: 42,
  TARGET_LOSS: 0.0001
}
Training samples: 480
Test samples    : 120 (non utilisés pour l'entraînement)
Epoch 1/10000 - Loss: 0.254247
Epoch 15/10000 - Loss: 0.000077
Perte cible 0.0001 atteinte à l'époque 15
Training: 62.497ms
Model saved to models/rps.json
```

```
$ npm run table -- rps          (extrait)
=== TEST ===
Input                                    | y_true           | y_pred           | err_%      | ok?
-----------------------------------------------------------------------------------------------------
scissors -> rock                         | 0                | 0                | 0.00       | OK
paper -> scissors                        | 2                | 2                | 0.00       | OK
…
=== Global ===
Train accuracy : 100.00 %
Test accuracy  : 100.00 %
```

```
$ npm run exec -- rps 1 0 0          (extrait)
Output: [0.000047,0.999936,0.000017]
```

Les classes sont numérotées 0 (pierre), 1 (feuille) et 2 (ciseaux) : contre la pierre, le réseau joue la feuille.

XOR, qui atteint maintenant la perte cible :

```
$ npm run train -- xor          (extrait)
Epoch 3000/10000 - Loss: 0.000234
Epoch 3394/10000 - Loss: 0.000100
Perte cible 0.0001 atteinte à l'époque 3394
```

Images (100 images, dont 20 réservées au test), avec la descente de gradient simple :

```
$ npm run train -- images --optimizer sgd --epochs 3000          (extrait)
Training samples: 80
Test samples    : 20 (non utilisés pour l'entraînement)
Epoch 1/3000 - Loss: 0.728013
Epoch 500/3000 - Loss: 0.007389
Epoch 1000/3000 - Loss: 0.002527
Epoch 2000/3000 - Loss: 0.001067
Epoch 3000/3000 - Loss: 0.000667
Training: 43.559s
```

```
$ npm run table -- images          (extrait)
=== TEST ===
orange_9_ok.jpg                          | 1.000000         | 0.000482         | 99.95      | KO
orange_58_ko.jpg                         | 0.000000         | 0.991127         | 99.11      | KO
orange_16_ok.jpg                         | 1.000000         | 0.219712         | 78.03      | KO
orange_94_ko.jpg                         | 0.000000         | 0.096271         | 9.63       | OK
…
=== Global ===
Train accuracy : 100.00 %
Test accuracy  : 50.00 %
```

La séparation montre ce que l'évaluation sur les images d'entraînement ne pouvait pas montrer : le réseau a appris par cœur ses 80 images, mais il ne reconnaît pas mieux que le hasard les 20 qu'il n'a jamais vues. Avec si peu d'images de 32 × 32 pixels en niveaux de gris, un réseau à une couche cachée ne généralise pas. Avec Adam au taux par défaut, la perte oscille sur ce jeu (entre 0,45 et 0,65 après 10 000 époques), comme dans le code de l'étape 8.

## Notes par étape

Ce que chaque tag `step-NN` contient d'inexact ou de piégeux. Ces tags ne sont pas modifiés ; la mention *corrigé dans `claude-NN`* indique où le point a été repris.

- **Étapes 3 et 4** : l'erreur de la couche cachée est calculée avec les poids de sortie *déjà mis à jour* (`this.output.backward(...)` est appelé avant leur lecture). L'étape 5 corrige ce point en copiant les poids avant la mise à jour.
- **Étape 4** : `exec` affiche `ReLU=` pour chaque neurone caché alors que la couche cachée est en tanh ; seul le libellé est faux.
- **Étapes 6 à 8** : le dossier `tmp/oranges/` doit exister, même vide, quel que soit le jeu de données. Le module `src/datasets/images.js` lit ce dossier dès son chargement, et `src/datasets/index.js` l'importe toujours. *Corrigé dans `claude-01`.*
- **Étapes 6 à 8** : `exec` redimensionne l'image avec le réglage par défaut de `sharp` (`cover`, qui recadre), alors que le jeu de données l'étire (`fit: "fill"`) ; l'image donnée à `exec` n'est donc pas préparée comme les images d'entraînement. *Corrigé dans `claude-03`.*
- **Étape 6** : aucun modèle entraîné n'est fourni ; il faut lancer `train` avant `table` et `exec`.
- **Étape 7** : l'exemple compare les optimiseurs à taux d'apprentissage égal (0,0025, la valeur par défaut). À ce taux et en 2000 époques, la descente de gradient simple n'a pas encore appris XOR ; avec les 10 000 époques par défaut, elle y parvient.
- **Étape 8** :
  - la séparation entraînement/test n'est pas utilisée : aucun jeu de données ne définit `indexEndTrain`, et les `testSamples` que prépare `rps` ne sont lus ni par `train` ni par `table`. *Corrigé dans `claude-01`* ;
  - l'entraînement ne s'arrête pas quand l'erreur atteint zéro, mais seulement si la perte totale devient négative (`totalLoss < 0`). Cela n'arrive qu'à cause de l'epsilon de `log(p + eps)` dans l'entropie croisée, quand les probabilités prédites sont très proches de 1. *Corrigé dans `claude-01`* ;
  - `rps` ne contient que 3 entrées différentes, répétées 200 fois puis mélangées. Les exemples de test sont donc les mêmes entrées que ceux d'entraînement : ils vérifient le mécanisme, pas la généralisation.
- **Classification binaire (étapes 5 à 8)** : le gradient `p − y` de l'entropie croisée avec sigmoïde est encore multiplié par la dérivée de la sigmoïde dans `Neuron.backward`. Le réseau apprend quand même, mais plus lentement. *Corrigé dans `claude-01`.*

## Licence

[MIT](LICENSE)
