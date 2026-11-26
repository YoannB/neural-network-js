# Réseau de neurones en JavaScript, étape par étape

Un réseau de neurones écrit en JavaScript pur (Node.js, sans bibliothèque d'apprentissage automatique), construit pas à pas : d'un neurone unique à un petit framework avec optimiseurs et classification multiclasse.

*A neural network written from scratch in plain JavaScript (Node.js), built step by step. Each step is a git tag, and the README of each tag describes that step with real run outputs. The text is in French.*

## Origine du projet

- **Le code** a été écrit en janvier 2026 avec l'aide d'une IA (ChatGPT). Une partie de l'étape 7 provient d'une version écrite avec Claude ; le commit de cette étape porte donc un co-auteur supplémentaire.
- **L'historique git** a été reconstitué en novembre 2026 à partir des dossiers de versions datés : chaque étape est un commit à la date de modification de ses fichiers, avec un tag.
- **Les README de chaque tag** ont été rédigés à ce moment-là avec Claude, d'où la ligne `Co-Authored-By` sur les commits. Les exemples d'exécution qu'ils montrent sont de vraies exécutions.

Le projet s'est amélioré d'étape en étape : le code des premières étapes peut contenir des erreurs corrigées plus tard, et l'étape 8 est la plus aboutie. Les tags restent des instantanés fidèles du projet ; ce qui est inexact ou piégeux dans chacun est indiqué plus bas, dans les [notes par étape](#notes-par-étape).

| Tag | Étape |
|---|---|
| `step-01-first-neuron` | Un neurone : y = w·x |
| `step-02-neuron-bias` | Un neurone avec biais : y = w·x + b |
| `step-03-neurons` | Un premier réseau 1-3-1 et la rétropropagation |
| `step-04-x-squared` | Approximation de x² |
| `step-05-mini-framework` | Mini-framework et classification binaire |
| `step-06-images` | Classification d'images |
| `step-07-optimizers` | Optimiseurs Adam et RMSprop |
| **`step-08-multiclass`** | **Classification multiclasse** ← version de `main` |

Pour passer d'une étape à l'autre : `git checkout <tag>`. Prérequis : Node.js 18 ou plus.

---

## Étape 8 — Classification multiclasse

Le réseau peut maintenant avoir plusieurs sorties et choisir une classe parmi plusieurs.

- **Softmax** sur la couche de sortie (`src/core/activations.js`, `src/core/layer.js`) : les sorties deviennent des probabilités dont la somme vaut 1.
- **Entropie croisée catégorielle** pour la tâche `multiclass-classification` ; l'erreur de chaque sortie est rétropropagée vers la couche cachée.
- Nouveau jeu de données `rps` (pierre-feuille-ciseaux) : pour chaque coup, encodé en vecteur à trois valeurs, le réseau doit donner le coup qui le bat.
- **Séparation entraînement / test préparée** : si un jeu de données définit `indexEndTrain`, `train` n'apprend que sur la première partie et `table` évalue les deux parties séparément. Aucun jeu de données fourni ne la définit encore (voir les notes par étape).
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

L'entraînement s'arrête ici à l'époque 63 : la boucle s'interrompt dès que la perte totale devient négative (voir les notes par étape).

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

## Notes par étape

Ce que chaque tag contient d'inexact ou de piégeux. Les tags ne sont pas modifiés ; ces points seront repris dans une version ultérieure.

- **Étapes 3 et 4** : l'erreur de la couche cachée est calculée avec les poids de sortie *déjà mis à jour* (`this.output.backward(...)` est appelé avant leur lecture). L'étape 5 corrige ce point en copiant les poids avant la mise à jour.
- **Étape 4** : `exec` affiche `ReLU=` pour chaque neurone caché alors que la couche cachée est en tanh ; seul le libellé est faux.
- **Étapes 6 à 8** : le dossier `tmp/oranges/` doit exister, même vide, quel que soit le jeu de données. Le module `src/datasets/images.js` lit ce dossier dès son chargement, et `src/datasets/index.js` l'importe toujours.
- **Étape 6** : aucun modèle entraîné n'est fourni ; il faut lancer `train` avant `table` et `exec`.
- **Étape 7** : l'exemple compare les optimiseurs à taux d'apprentissage égal (0,0025, la valeur par défaut). À ce taux et en 2000 époques, la descente de gradient simple n'a pas encore appris XOR ; avec les 10 000 époques par défaut, elle y parvient.
- **Étape 8** :
  - la séparation entraînement/test n'est pas utilisée : aucun jeu de données ne définit `indexEndTrain`, et les `testSamples` que prépare `rps` ne sont lus ni par `train` ni par `table` ;
  - l'entraînement ne s'arrête pas quand l'erreur atteint zéro, mais seulement si la perte totale devient négative (`totalLoss < 0`). Cela n'arrive qu'à cause de l'epsilon de `log(p + eps)` dans l'entropie croisée, quand les probabilités prédites sont très proches de 1 ;
  - `rps` ne contient que 3 entrées différentes, répétées 200 fois puis mélangées.
- **Classification binaire (étapes 5 à 8)** : le gradient `p − y` de l'entropie croisée avec sigmoïde est encore multiplié par la dérivée de la sigmoïde dans `Neuron.backward`. Le réseau apprend quand même, mais plus lentement.

## Licence

[MIT](LICENSE)
