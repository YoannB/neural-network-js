# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

L'historique git a été reconstitué à partir des versions successives du projet, à leurs dates d'origine (janvier 2026). Chaque étape a son tag ; le README de chaque tag présente l'étape correspondante, avec des exemples d'exécution réels.

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| **`step-04-x-squared`** | **Approximation de x²** ← ce tag |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 4 — Approximation de x²

Le réseau apprend une fonction non linéaire : `y = x²` sur l'intervalle [0, 5], à partir de 51 points (`data/dataset.json`).

- couche cachée de 12 neurones en **tanh**, sortie **linéaire** (`src/activations.js` ajoute aussi leaky ReLU) ;
- initialisation des poids à l'échelle `√(2 / nombre d'entrées)` ;
- **gradients écrêtés** (limités à ±5) et contrôle des valeurs non finies ;
- **normalisation** des entrées et des sorties, calculée à partir du jeu de données et sauvegardée avec le modèle ;
- paramètres en ligne de commande : `--epochs`, `--neurons`, `--lr`, `--noise` (bruit ajouté aux entrées) ;
- `npm run table` compare la valeur attendue et la valeur prédite pour chaque point ;
- `npm run exec` affiche aussi l'activation de chaque neurone caché.

## Lancer

```sh
npm run exec -- 2.5
npm run table
npm run train -- --epochs 10000 --neurons 12 --lr 0.003
```

## Exemples d'exécution

Avec le modèle fourni :

```
$ npm run exec -- 2.5
x = 2.5
Activations cachées :
  Neurone 0 | z=4.0778 | ReLU=0.9994
  Neurone 1 | z=-0.6901 | ReLU=-0.5980
  Neurone 2 | z=-0.5296 | ReLU=-0.4851
  Neurone 3 | z=0.7274 | ReLU=0.6215
  Neurone 4 | z=0.9147 | ReLU=0.7234
  Neurone 5 | z=0.1988 | ReLU=0.1962
  Neurone 6 | z=-1.5080 | ReLU=-0.9066
  Neurone 7 | z=-2.4685 | ReLU=-0.9857
  Neurone 8 | z=-0.9326 | ReLU=-0.7318
  Neurone 9 | z=1.3073 | ReLU=0.8636
  Neurone 10 | z=-1.3334 | ReLU=-0.8701
  Neurone 11 | z=0.9686 | ReLU=0.7481
Sortie finale y ≈ 6.2546
```

```
$ npm run table          (extrait)
x      | y_true     | y_pred
--------------------------------------------------------------------------------
0.00   | 0.00       | -0.0732
0.50   | 0.25       | 0.3887
1.00   | 1.00       | 1.0993
1.50   | 2.25       | 2.2346
2.00   | 4.00       | 3.9440
2.50   | 6.25       | 6.2546
3.00   | 9.00       | 9.0634
3.50   | 12.25      | 12.2576
4.00   | 16.00      | 15.9210
4.50   | 20.25      | 20.3087
5.00   | 25.00      | 25.0751
```

Entraînement :

```
$ npm run train
Training config:
{ EPOCHS: 10000, NEURONS: 12, LR: 0.003, NOISE: 0 }
Epoch 500/10000 - Loss: 0.011159
Epoch 1000/10000 - Loss: 0.001892
Epoch 1500/10000 - Loss: 0.001670
…
Epoch 9000/10000 - Loss: 0.001035
Epoch 9500/10000 - Loss: 0.000982
Epoch 10000/10000 - Loss: 0.000933
Model saved to model/model.json
```
