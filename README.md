# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

L'historique git a été reconstitué à partir des versions successives du projet, à leurs dates d'origine (janvier 2026). Chaque étape a son tag ; le README de chaque tag présente l'étape correspondante, avec des exemples d'exécution réels.

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| **`step-03-neurons`** | **Un premier réseau 1-3-1 et la rétropropagation** ← ce tag |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 3 — Un premier réseau 1-3-1 et la rétropropagation

Le neurone unique devient un réseau : une entrée, une couche cachée de trois neurones, un neurone de sortie, avec l'activation ReLU. Le code est découpé en trois classes :

- `src/neuron.js` : un neurone à plusieurs entrées (`forward`, `backward`)
- `src/layer.js` : une couche de neurones
- `src/network.js` : le réseau 1 → 3 → 1 ; l'erreur de sortie est rétropropagée vers la couche cachée en la multipliant par les poids de sortie
- `src/activations.js` : ReLU et sa dérivée

Le jeu de données reste `y = 3x`. Le modèle sauvegardé contient maintenant tous les poids et biais du réseau.

## Lancer

```sh
npm run exec -- 4
npm run train -- 1000
npm run exec -- 7
```

## Exemples d'exécution

```
$ npm run exec -- 4
Input : 4
Output: 12.0000
```

```
$ npm run train
Epoch 100/1000 - Loss: 0.0182
Epoch 200/1000 - Loss: 0.0000
Epoch 300/1000 - Loss: 0.0000
Epoch 400/1000 - Loss: 0.0000
Epoch 500/1000 - Loss: 0.0000
Epoch 600/1000 - Loss: 0.0000
Epoch 700/1000 - Loss: 0.0000
Epoch 800/1000 - Loss: 0.0000
Epoch 900/1000 - Loss: 0.0000
Epoch 1000/1000 - Loss: 0.0000
Entraînement terminé.

$ npm run exec -- 7
Input : 7
Output: 21.0000
```

Les poids de départ sont aléatoires : d'un entraînement à l'autre, la courbe d'erreur varie.
