# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

L'historique git a été reconstitué à partir des versions successives du projet, à leurs dates d'origine (janvier 2026). Chaque étape a son tag ; le README de chaque tag présente l'étape correspondante, avec des exemples d'exécution réels.

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| **`step-02-neuron-bias`** | **Un neurone avec biais : y = w·x + b** ← ce tag |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 2 — Un neurone avec biais : y = w·x + b

Le neurone reçoit un biais `b` en plus de son poids : il peut maintenant représenter une droite qui ne passe pas par l'origine. Les deux paramètres sont corrigés à chaque exemple (gradient `2 × erreur × x` pour le poids, `2 × erreur` pour le biais). Le jeu de données reste `y = 3x`, et l'entraînement retrouve `w = 3` et `b = 0`.

## Lancer

```sh
npm run exec -- 4        # prédiction avec le modèle fourni
npm run train -- 1000
npm run exec -- 10
```

## Exemples d'exécution

```
$ npm run exec -- 4
Input : 4
Output: 12.00
```

```
$ npm run train
Epoch 100/1000 | Loss=0.0180 | w=2.9689 | b=0.1318
Epoch 200/1000 | Loss=0.0005 | w=2.9949 | b=0.0217
Epoch 300/1000 | Loss=0.0000 | w=2.9992 | b=0.0036
Epoch 400/1000 | Loss=0.0000 | w=2.9999 | b=0.0006
Epoch 500/1000 | Loss=0.0000 | w=3.0000 | b=0.0001
Epoch 600/1000 | Loss=0.0000 | w=3.0000 | b=0.0000
Epoch 700/1000 | Loss=0.0000 | w=3.0000 | b=0.0000
Epoch 800/1000 | Loss=0.0000 | w=3.0000 | b=0.0000
Epoch 900/1000 | Loss=0.0000 | w=3.0000 | b=0.0000
Epoch 1000/1000 | Loss=0.0000 | w=3.0000 | b=0.0000
Entraînement terminé.

$ npm run exec -- 10
Input : 10
Output: 30.00
```
