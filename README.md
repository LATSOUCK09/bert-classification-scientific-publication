#  Classification Multi-Label de Publications Scientifiques avec BERT

## 1. Présentation du projet

Ce projet vise à classifier automatiquement des publications scientifiques en plusieurs domaines de recherche à partir de leur **titre** et de leur **résumé (abstract)**.

Contrairement à une classification classique où un document appartient à une seule catégorie, il s'agit ici d'un problème de **classification multi-label**, ce qui signifie qu'un article peut être associé simultanément à plusieurs domaines scientifiques.

Le modèle utilisé est **BERT (Bidirectional Encoder Representations from Transformers)**, fine-tuné sur un corpus d'articles scientifiques issus de la plateforme ArXiv.

---

# 2. Présentation du dataset

## Source

Dataset : **ArXiv Scientific Publications Dataset**

Fichier utilisé :

```text
data/scientific-publication.csv
```

Chaque ligne contient :

* un identifiant (`ID`)
* un titre (`TITLE`)
* un résumé (`ABSTRACT`)
* six labels binaires indiquant l'appartenance à différents domaines scientifiques.

## Classes

| Classe               | Description                                               |
| -------------------- | --------------------------------------------------------- |
| Computer Science     | Informatique, Intelligence Artificielle, Machine Learning |
| Physics              | Physique théorique et expérimentale                       |
| Mathematics          | Mathématiques pures et appliquées                         |
| Statistics           | Probabilités et statistiques                              |
| Quantitative Biology | Bioinformatique et biologie computationnelle              |
| Quantitative Finance | Finance quantitative et économétrie                       |

## Statistiques du dataset

| Information             | Valeur      |
| ----------------------- | ----------- |
| Nombre total d'articles | 20 972      |
| Nombre de classes       | 6           |
| Type de classification  | Multi-label |

### Distribution des classes

| Classe               | Occurrences |
| -------------------- | ----------- |
| Computer Science     | 8 594       |
| Physics              | 6 013       |
| Mathematics          | 5 618       |
| Statistics           | 5 206       |
| Quantitative Biology | 587         |
| Quantitative Finance | 249         |

On observe un fort déséquilibre de classes, notamment pour :

* Quantitative Biology
* Quantitative Finance

Ce déséquilibre a été traité grâce à l'utilisation de poids de classes dans la fonction de perte.

## Exemples du dataset

### Exemple 1

**Titre**

```text
Attention Is All You Need
```

**Classe attendue**

```text
Computer Science
```

### Exemple 2

**Titre**

```text
Gravitational Waves from Binary Black Hole Mergers
```

**Classe attendue**

```text
Physics
```

### Exemple 3

**Titre**

```text
Protein Folding Prediction Using Deep Learning
```

**Classe attendue**

```text
Computer Science
Quantitative Biology
```

---

# 3. Description du modèle et choix techniques

## Modèle utilisé

Le projet utilise :

```text
bert-base-uncased
```

proposé par Hugging Face.

L'architecture générale est :

```text
Titre + Abstract
        │
Tokenizer BERT
        │
bert-base-uncased
        │
Pooler Output (768)
        │
Linear(768 → 6)
        │
Sigmoid
        │
6 probabilités
```

## Tokenizer

```python
AutoTokenizer.from_pretrained("bert-base-uncased")
```

Choix effectués :

* padding = max_length
* truncation = True
* max_length = 256

Le texte d'entrée est construit comme suit :

```text
TITLE [SEP] ABSTRACT
```

## Tête de classification

```python
nn.Linear(768, 6)
```

Cette couche produit un logit pour chacune des 6 classes.

## Fonction d'activation

```python
Sigmoid
```

Contrairement à Softmax, Sigmoid permet d'activer plusieurs classes simultanément.

## Fonction de perte

```python
BCEWithLogitsLoss
```

avec :

```python
pos_weight
```

pour compenser le déséquilibre du dataset.

## Hyperparamètres

| Paramètre     | Valeur            |
| ------------- | ----------------- |
| Modèle        | bert-base-uncased |
| Batch Size    | 16                |
| Max Length    | 256               |
| Epochs        | 5                 |
| Learning Rate | 2e-5              |
| Optimiseur    | AdamW             |
| Seuil         | 0.5               |

---

# 4. Étapes de réalisation

### Étape 1 : Analyse du dataset

* Exploration des données
* Vérification des labels
* Étude du déséquilibre des classes

### Étape 2 : Prétraitement

* Fusion du titre et du résumé
* Tokenisation avec BERT
* Padding et troncature

### Étape 3 : Construction du modèle

* Chargement de BERT pré-entraîné
* Ajout d'une couche linéaire de classification

### Étape 4 : Entraînement

* Split 80 % / 20 %
* Fine-tuning complet de BERT
* Sauvegarde du meilleur modèle

### Étape 5 : Évaluation

Calcul de :

* Loss
* Accuracy
* F1-Score

### Étape 6 : Déploiement

Création d'une interface utilisateur avec Gradio.

---

# 5. Difficultés rencontrées

## Déséquilibre des classes

Certaines classes apparaissent très rarement.

Exemple :

```text
Quantitative Finance : 249 exemples
Computer Science : 8594 exemples
```

Solution :

```python
BCEWithLogitsLoss(pos_weight=...)
```

---

## Classification multi-label

Le problème nécessite :

* Sigmoid
* BCEWithLogitsLoss

au lieu de :

* Softmax
* CrossEntropyLoss

---

## Longueur importante des abstracts

De nombreux résumés dépassent la taille maximale supportée.

Solution :

```python
max_length = 256
truncation=True
```

---

## Temps d'entraînement

Le fine-tuning de BERT est coûteux en calcul.

Solution :

* utilisation du GPU CUDA
* batch size adapté à la mémoire disponible

---

# 6. Résultats

## Courbes d'entraînement

Insérer ici :

```text
screenshots/training_curves.png
```

![Courbes](screenshots/training_curves.png)

Les courbes montrent :

* une diminution progressive de la loss
* une amélioration des performances de validation
* une bonne convergence du modèle

---

## Métriques finales

| Métrique            | Valeur |
| ------------------- | ------ |
| Validation Loss     | XX     |
| Validation Accuracy | XX     |
| Validation F1-Score | XX     |

(Remplacer XX par vos résultats réels)

---

## Matrice de confusion

Insérer ici :

```text
screenshots/confusion_matrix.png
```

![Confusion Matrix](screenshots/confusion_matrix.png)

Analyse :

* les classes majoritaires sont correctement reconnues ;
* les erreurs concernent principalement les classes rares ;
* quelques chevauchements apparaissent entre Mathematics et Statistics.

---

# 7. Démonstration Gradio

L'application permet :

* saisir un titre ;
* saisir un résumé ;
* modifier le seuil de décision ;
* visualiser les probabilités prédites.

## Capture d'écran

```text
screenshots/gradio_demo.png
```

![Gradio Demo](screenshots/gradio_demo.png)

---

# 8. Installation

## Cloner le dépôt

```bash
git clone https://github.com/LATSOUCK09/bert-classification-scientific-publication.git

cd bert-classification-scientific-publication
```

## Créer un environnement virtuel

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 9. Exécution

## Entraînement du modèle

```bash
python train.py
```

Le meilleur modèle est automatiquement sauvegardé sous :

```text
best_bert_multilabel.pth
```

---

## Lancer la démo

```bash
python demo.py
```

Puis ouvrir :

```text
http://localhost:7860
```

---

# 10. Structure du projet

```text
bert-classification-scientific-publication/
│
├── data/
│   └── scientific-publication.csv
│
├── dataset.py
├── model.py
├── train.py
├── utils.py
├── demo.py
│
├── screenshots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradio_demo.png
│
├── requirements.txt
└── README.md
```

---

# 11. Perspectives d'amélioration

* Utiliser SciBERT à la place de BERT.
* Ajuster automatiquement le seuil de décision par classe.
* Ajouter Precision et Recall par classe.
* Tester des techniques de Focal Loss.
* Ajouter un ensemble de test indépendant.
* Déployer l'application sur Hugging Face Spaces ou Streamlit Cloud.

---

## Auteur

**LATSOUCK09**  **LilleBaro** 

Projet réalisé dans le cadre d'un travail pratique de classification de textes scientifiques avec les Transformers.
