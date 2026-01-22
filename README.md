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
| `step-06-images` | Classification d'images |
| **`step-07-optimizers`** | **Optimiseurs Adam et RMSprop** ← ce tag |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 7 — Optimiseurs Adam et RMSprop

Le framework gagne plusieurs façons de mettre à jour les poids, et chaque jeu de données a son propre modèle.

- **Optimiseurs** (`src/core/neuron.js`), au choix avec `--optimizer` :
  - `sgd` : descente de gradient simple, comme aux étapes précédentes ;
  - `rmsprop` : le pas est divisé par la moyenne glissante du carré des gradients ;
  - `adam` (par défaut) : moyennes glissantes du gradient et de son carré, avec correction du biais.
- **Sauvegarde** : `Network.toJSON()` ; chaque modèle est écrit dans `models/<jeu de données>.json`, et `train`, `exec` et `table` prennent le nom du jeu de données en premier argument.
- `forward()` est commun à `predict` et `debugPredict`, et les entrées sont toujours normalisées.
- `Layer.getWeightStats()` donne la moyenne, l'écart-type, le minimum et le maximum des poids d'une couche.
- Nouveau jeu de données `linear` (y = 2x − 1) ; `npm run reset` vide le dossier `models/`.

## Lancer

```sh
npm install
mkdir -p tmp/oranges          # dossier du jeu d'images, voir l'étape 6
npm run train -- xor --epochs 2000 --optimizer adam
npm run table -- xor
npm run exec -- xor 1 0
```

## Exemples d'exécution

XOR en 2000 époques avec chacun des trois optimiseurs (taux d'apprentissage par défaut 0,0025) :

```
$ npm run train -- xor --epochs 2000 --optimizer sgd
Epoch 500/2000 - Loss: 0.628462
Epoch 1000/2000 - Loss: 0.599765
Epoch 1500/2000 - Loss: 0.569346
Epoch 2000/2000 - Loss: 0.534552
Training: 30.693ms

$ npm run train -- xor --epochs 2000 --optimizer rmsprop
Epoch 500/2000 - Loss: 0.286819
Epoch 1000/2000 - Loss: 0.005074
Epoch 1500/2000 - Loss: 0.000030
Epoch 2000/2000 - Loss: 0.000006
Training: 32.026ms

$ npm run train -- xor --epochs 2000 --optimizer adam
Epoch 500/2000 - Loss: 0.217452
Epoch 1000/2000 - Loss: 0.057698
Epoch 1500/2000 - Loss: 0.022529
Epoch 2000/2000 - Loss: 0.010248
Training: 47.71ms
```

```
$ npm run table -- xor
Input                                    | y_true       | y_pred       | err_%      | ok?
--------------------------------------------------------------------------------------------
[0,0]                                    | 0.000000     | 0.008583     | 0.86       | OK
[0,1]                                    | 1.000000     | 0.989015     | 1.10       | OK
[1,0]                                    | 1.000000     | 0.989990     | 1.00       | OK
[1,1]                                    | 0.000000     | 0.011146     | 1.11       | OK

Résumé:
Task           : binary-classification
Samples        : 4
Accuracy       : 100.00 %
False positives: 0
False negatives: 0

$ npm run exec -- xor 1 0
Read ./models/xor.json
Debug predict : [ 1, 0 ]

Hidden:
Neuron 0 | z=1.48286 | a=0.902003
…
Neuron 9 | z=-2.461344 | a=-0.985546

Output: 0.98999
```

```
$ npm run train -- linear --epochs 2000 --optimizer adam
…
Epoch 2000/2000 - Loss: 0.000004
Model saved to models/linear.json

$ npm run exec -- linear 5
…
Output: 8.81323
```
