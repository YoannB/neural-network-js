# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

L'historique git a été reconstitué à partir des versions successives du projet, à leurs dates d'origine (janvier 2026). Chaque étape a son tag ; le README de chaque tag présente l'étape correspondante, avec des exemples d'exécution réels.

| Tag | Étape |
|---|---|
| **`step-01-first-neuron`** | **Un neurone : y = w·x** ← ce tag |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 1 — Un neurone : y = w·x

Un seul neurone, avec un seul poids `w`, apprend la relation `y = 3x` à partir de cinq exemples (`data/dataset.json`). L'apprentissage se fait par descente de gradient sur l'erreur quadratique : à chaque exemple, le poids est corrigé de `learningRate × 2 × erreur × x`.

- `src/neuron.js` : le neurone (prédiction, erreur, mise à jour du poids)
- `src/train.js` : entraînement, puis sauvegarde du poids dans `model/model.json`
- `src/exec.js` : prédiction avec le modèle sauvegardé
- `src/reset.js` : remet un poids aléatoire

## Lancer

```sh
npm run exec -- 4        # prédiction avec le modèle fourni
npm run train -- 1000    # entraînement (nombre d'époques, 1000 par défaut)
npm run exec -- 10
```

## Exemples d'exécution

Avec le modèle fourni :

```
$ npm run exec -- 4
Input: 4
Output: 12.00
```

Entraînement :

```
$ npm run train
Epoch 100/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 200/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 300/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 400/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 500/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 600/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 700/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 800/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 900/1000 - Loss: 0.0000 - Weight: 3.0000
Epoch 1000/1000 - Loss: 0.0000 - Weight: 3.0000
Entraînement terminé.

$ npm run exec -- 10
Input: 10
Output: 30.00
```
