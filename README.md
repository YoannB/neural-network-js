# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

## Branche `claude-improvement`

Tu es sur la branche `claude-improvement` : c'est la reprise du projet par Claude, à partir de l'étape 8. Elle montre comment le code peut évoluer vers un mini framework testé, puis vers TypeScript.

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
| **`claude-01-fiabilite`** | **Corrections, graine aléatoire, séparation entraînement / test, tests** |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## claude-01 — Fiabilité

Cette étape corrige les points relevés dans les [notes par étape](#notes-par-étape) et rend les résultats vérifiables.

- **Graine aléatoire** (`src/core/random.js`) : l'initialisation des poids, le mélange des données et le bruit dépendent de `--seed` (42 par défaut). Deux entraînements avec la même graine donnent le même modèle. La graine est enregistrée dans le modèle.
- **Séparation entraînement / test** (`src/datasets/split.js`) : `rps` et `images` réservent 20 % de leurs exemples au test. `train` n'apprend que sur les 80 % restants et `table` évalue les deux parties séparément.
- **Gradient de la sortie sigmoïde** : en classification binaire, le gradient `p − y` n'est plus multiplié par la dérivée de la sigmoïde. Sur XOR, la perte atteinte en 3000 époques est environ dix fois plus basse qu'avant.
- **Critère d'arrêt** (`src/core/training.js`) : l'entraînement s'arrête quand la perte moyenne descend sous `--target-loss` (0,0001 par défaut, 0 pour désactiver). Les entropies croisées ne peuvent plus être négatives.
- **Jeux de données chargés à la demande** : le dossier `tmp/oranges/` n'est plus nécessaire pour les jeux autres que `images`.
- **Sorties multiclasses** : `table` les lit avec `predict`, et `exec` affiche toutes les probabilités.
- **Tests** : `npm test` lance les tests avec `node:test`, le module de test intégré à Node, sans dépendance supplémentaire. Ils vérifient notamment la rétropropagation par différences finies pour les trois tâches.

## Lancer

```sh
npm install
npm test
npm run train -- rps
npm run table -- rps
npm run exec -- rps 1 0 0
```

Options de `train` : `--epochs`, `--neurons`, `--lr`, `--noise`, `--optimizer` (`sgd`, `adam`, `rmsprop`), `--seed`, `--target-loss`.

Sous Windows PowerShell 5.1, `npm run train -- rps --epochs 500` peut perdre le `--`, et les options ne sont alors pas transmises. Dans ce cas, appelle les scripts directement : `node src/train.js rps --epochs 500`.

Jeux de données : `linear`, `square`, `xor`, `primes`, `rps`, `images`.

### Préparer le jeu `images`

Le jeu `images` lit les fichiers `.jpg` du dossier `tmp/oranges/`, qui n'est pas versionné. L'étiquette vient du nom du fichier : `_ok` dans le nom pour une orange (classe 1), sinon une autre image (classe 0).

1. Crée une clé d'API gratuite sur [Pixabay](https://pixabay.com/api/docs/) et mets-la dans un fichier `.env.local` à la racine du projet. Ce fichier n'est pas versionné : le `.gitignore` exclut `.env` et `.env.*`.

   ```
   PIXABAY_API_KEY=ta_clé
   ```

2. Télécharge 100 images. Node ne lit pas les fichiers `.env` tout seul : l'option `--env-file` (Node 20.6 ou plus) s'en charge.

   ```sh
   node --env-file=.env.local scripts/download-oranges.js
   ```

   Sans cette option, il faut définir la variable `PIXABAY_API_KEY` dans le terminal.

3. Trie les images à la main. Le script les nomme `orange_1.jpg`, `orange_2.jpg`… : sans `_ok` dans le nom, elles seraient toutes de la classe 0 et le réseau n'aurait rien à apprendre. Renomme chaque fichier :
   - `orange_12_ok.jpg` pour une vraie orange ;
   - `orange_13_ko.jpg` pour une image que le réseau doit rejeter.

4. Entraîne puis évalue. Avec la descente de gradient simple, la perte baisse régulièrement ; avec Adam, elle oscille sur ce jeu.

   ```sh
   node src/train.js images --optimizer sgd --epochs 3000
   node src/table.js images
   node src/exec.js images chemin/vers/une-image.jpg
   ```

## Exemples d'exécution

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

Ce que chaque tag `step-NN` contient d'inexact ou de piégeux. Ces tags ne sont pas modifiés ; la mention *corrigé dans `claude-01`* indique où le point a été repris.

- **Étapes 3 et 4** : l'erreur de la couche cachée est calculée avec les poids de sortie *déjà mis à jour* (`this.output.backward(...)` est appelé avant leur lecture). L'étape 5 corrige ce point en copiant les poids avant la mise à jour.
- **Étape 4** : `exec` affiche `ReLU=` pour chaque neurone caché alors que la couche cachée est en tanh ; seul le libellé est faux.
- **Étapes 6 à 8** : le dossier `tmp/oranges/` doit exister, même vide, quel que soit le jeu de données. Le module `src/datasets/images.js` lit ce dossier dès son chargement, et `src/datasets/index.js` l'importe toujours. *Corrigé dans `claude-01`.*
- **Étape 6** : aucun modèle entraîné n'est fourni ; il faut lancer `train` avant `table` et `exec`.
- **Étape 7** : l'exemple compare les optimiseurs à taux d'apprentissage égal (0,0025, la valeur par défaut). À ce taux et en 2000 époques, la descente de gradient simple n'a pas encore appris XOR ; avec les 10 000 époques par défaut, elle y parvient.
- **Étape 8** :
  - la séparation entraînement/test n'est pas utilisée : aucun jeu de données ne définit `indexEndTrain`, et les `testSamples` que prépare `rps` ne sont lus ni par `train` ni par `table`. *Corrigé dans `claude-01`* ;
  - l'entraînement ne s'arrête pas quand l'erreur atteint zéro, mais seulement si la perte totale devient négative (`totalLoss < 0`). Cela n'arrive qu'à cause de l'epsilon de `log(p + eps)` dans l'entropie croisée, quand les probabilités prédites sont très proches de 1. *Corrigé dans `claude-01`* ;
  - `rps` ne contient que 3 entrées différentes, répétées 200 fois puis mélangées. Les exemples de test sont donc les mêmes entrées que ceux d'entraînement : ils vérifient le mécanisme, pas la généralisation.
- **Classification binaire (étapes 5 à 8)** : le gradient `p − y` de l'entropie croisée avec sigmoïde est encore multiplié par la dérivée de la sigmoïde dans `Neuron.backward`. Le réseau apprend quand même, mais plus lentement. *Corrigé dans `claude-01`.*

## Licence

[MIT](LICENSE)
