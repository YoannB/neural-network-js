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
| **`step-05-mini-framework`** | **Mini-framework et classification binaire** ← ce tag |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| `step-08-multiclass` | Classification multiclasse |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 5 — Mini-framework et classification binaire

Le code est réorganisé pour servir à plusieurs problèmes :

- `src/core/` : le réseau (neurone, couche, réseau, activations) ; les entrées sont maintenant des vecteurs, avec contrôle des dimensions ;
- `src/datasets/` : les jeux de données, choisis par leur nom en ligne de commande :
  - `square` : y = x² (régression),
  - `xor` : le OU exclusif (classification binaire),
  - `primes` : un nombre de 0 à 13, donné en binaire sur 5 bits, est-il premier ? (classification binaire) ;
- **classification binaire** : sortie sigmoïde et entropie croisée binaire ; le type de tâche est enregistré dans le modèle ;
- `exec` accepte plusieurs valeurs (`1 0`) ou un tableau JSON (`[1,0]`).

## Lancer

```sh
npm run table                         # modèle fourni : nombres premiers
npm run exec -- 0 1 1 0 1             # 13 en binaire
npm run train -- xor --epochs 5000 --lr 0.1
npm run table
npm run exec -- 1 0
npm run train -- square
npm run exec -- 3
```

## Exemples d'exécution

Avec le modèle fourni (nombres premiers) :

```
$ npm run table
Input                          | y_true       | y_pred
----------------------------------------------------------------------
[0,0,0,0,0]                    | 0.000000     | 0.003639
[0,0,0,0,1]                    | 0.000000     | 0.053864
[0,0,0,1,0]                    | 1.000000     | 0.914373
[0,0,0,1,1]                    | 1.000000     | 0.994995
[0,0,1,0,0]                    | 0.000000     | 0.013257
[0,0,1,0,1]                    | 1.000000     | 0.966516
[0,0,1,1,0]                    | 0.000000     | 0.073637
[0,0,1,1,1]                    | 1.000000     | 0.992434
[0,1,0,0,0]                    | 0.000000     | 0.000180
[0,1,0,0,1]                    | 0.000000     | 0.033000
[0,1,0,1,0]                    | 0.000000     | 0.051689
[0,1,0,1,1]                    | 1.000000     | 0.972385
[0,1,1,0,0]                    | 0.000000     | 0.006872
[0,1,1,0,1]                    | 1.000000     | 0.943000

$ npm run exec -- 0 1 1 0 1
Model dataset: primes
Expected input size: 5

Input: [ 0, 1, 1, 0, 1 ]
Activations cachées :
  Neurone 0 | z=0.258196 | activation=0.252607
  …
  Neurone 11 | z=0.425374 | activation=0.401448

Sortie finale y ≈ 0.943000
```

XOR :

```
$ npm run train -- xor --epochs 5000 --lr 0.1
Dataset: xor
Samples: 4
Input size: 2
Output size: 1
Task: binary-classification
Training config:
{ EPOCHS: 5000, NEURONS: 12, LR: 0.1, NOISE: 0 }
Epoch 500/5000 - Loss: 0.069761
Epoch 1000/5000 - Loss: 0.039118
…
Epoch 4500/5000 - Loss: 0.012242
Epoch 5000/5000 - Loss: 0.011282
Model saved to model/model.json

$ npm run table
Input                          | y_true       | y_pred
----------------------------------------------------------------------
[0,0]                          | 0.000000     | 0.002994
[0,1]                          | 1.000000     | 0.987098
[1,0]                          | 1.000000     | 0.986627
[1,1]                          | 0.000000     | 0.015533

$ npm run exec -- 1 0
…
Sortie finale y ≈ 0.986627
```

x² avec le même framework :

```
$ npm run train -- square
…
Epoch 10000/10000 - Loss: 0.000949
Model saved to model/model.json

$ npm run exec -- 3
…
Sortie finale y ≈ 9.041356
```
