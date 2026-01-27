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
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| **`step-08-multiclass`** | **Classification multiclasse** ← ce tag |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 8 — Classification multiclasse

Le réseau peut maintenant avoir plusieurs sorties et choisir une classe parmi plusieurs.

- **Softmax** sur la couche de sortie (`src/core/activations.js`, `src/core/layer.js`) : les sorties deviennent des probabilités dont la somme vaut 1.
- **Entropie croisée catégorielle** pour la tâche `multiclass-classification` ; l'erreur de chaque sortie est rétropropagée vers la couche cachée.
- Nouveau jeu de données `rps` (pierre-feuille-ciseaux) : pour chaque coup, encodé en vecteur à trois valeurs, le réseau doit donner le coup qui le bat.
- **Séparation entraînement / test** : un jeu de données peut définir `indexEndTrain` ; `train` n'apprend que sur la première partie et `table` évalue les deux parties séparément.
- `table` affiche la classe attendue et la classe prédite pour la classification multiclasse.

## Lancer

```sh
npm install
mkdir -p tmp/oranges          # dossier du jeu d'images, voir l'étape 6
npm run train -- rps --epochs 500
npm run table -- rps
```

Les autres jeux de données (`square`, `xor`, `primes`, `linear`, `images`) fonctionnent comme à l'étape 7.

## Exemples d'exécution

```
$ npm run train -- rps --epochs 500
Dataset    : rps
Model name : rps
Samples    : 480
Input size : 3
Output size: 3
Task       : multiclass-classification
Training config: { EPOCHS: 500, NEURONS: 10, LR: 0.0025, NOISE: 0, OPTIMIZER: 'adam' }
Training samples: 480
Epoch 1/500 - Loss: 0.319506
Epoch 63/500 - Loss: -0.000000
Training: 231.811ms
Model saved to models/rps.json
```

L'entraînement s'arrête dès que l'erreur arrondie atteint zéro (ici à l'époque 63).

```
$ npm run table -- rps          (extrait)
=== DATASET ===
Input                                    | y_true           | y_pred           | err_%      | ok?
-----------------------------------------------------------------------------------------------------
paper -> scissors                        | 2                | 2                | 0.00       | OK
rock -> paper                            | 1                | 1                | 0.00       | OK
…
Résumé (DATASET) :
Task           : multiclass-classification
Samples        : 480
Accuracy       : 100.00 %
```

Les classes sont numérotées 0 (pierre), 1 (feuille) et 2 (ciseaux).
