# Mini projet - Alexio BEC

## Installation

En plus des bibliothèques habituelles en Machine Learning, nous avons choisi d'utiliser la bibliothèque plotly pour afficher les graphes, ce qui permet de les manipuler (par exemple tourner pour des graphes 3D) puis de les enregistrer en format png très facilement.

Pour installer la bibliothèque il faut lancer la commande **pip install plotly**

## Contenu du git

* Les données sont dans le fichier imports-exports-commerciaux.csv
* Le code est disponible en jupyter notebook ou en fichier python (le code est le même mais il y a quelques affichages en plus dans le jupyter notebook)
## Données

Le jeu de données que j'ai choisi contient les données des échanges d'énergie entre la France et d'autres pays. Il est disponible à cette adresse : https://www.data.gouv.fr/fr/datasets/imports-et-exports-commerciaux-2005-a-2021/

Les données sont composées de plusieurs champs comme l'importation et l'exportation d'énergie française.

Il y a une valeur par heure tous les jours entre 2005 et 2021.

## Objectifs

L'objectif que j'ai fixé est de prédire l'exportation d'énergie sur un jour entier en prenant comme entrée les jours précédents.

Pour cela j'ai choisi d'utiliser un modèle composé d'une série de CNN avec un champ récepteur de 7 jours car je suis parti du principe qu'il y aurait une périodicité en fonction de la semaine, par exemple les données des lunids se ressembleront. Après les couches de CNN, j'ai déployé des couches de RNN.

## Preprocessing

Après avoir gardé que les champs intéressants et remis en forme la date,  comme les données étaient désordonnées dans le fichier csv, j'ai dû vérifier qu'elles étaient bien complètes pour pouvoir avoir une séquence entière. J'ai d'abord essayé de générer chaque date entre 2005 et 2021 et vérifier que la date se trouvait bien dans les données, cependant le nombre de date en question est environ de 150 000 et cette technique, codée naïvement, étant en O(n²), cela prenait trop de temps.

C'est pour cela que j'ai opté pour une autre stratégie, compter le nombre d'élément, vérifier que ce soit bien celui attendu et s'assurer qu'il n'y ait pas de doublons dans les dates. 

Cela a bien fonctionné, les données étient bien complètes.

Pour formater les données, je pensais d'abord, en utilisant chaque valeur heure par heure, empiler un CNN avec un champ récepteur de 24 pour avoir une journée, puis un de 7 pour obtenir une semaine. Cependant, je me suis dit que ce serait plus efficace de n'utiliser que les CNN avec un champ récepteur de la semaine mais qui ne prennent pas les valeur heure par heure mais des vecteurs des 24 valeurs de la journée, cela facilite aussi l'implémentation des RNN et la sortie d'un vecteur decrivant un jour.


## Présentation du modèle

Le modèle est composé d'une série de couches de CNN à taille de noyau variable, j'adapte la profondeur pour avoir un champ récepteur de 7 avec un stride de 1.

![CNNs](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/CNN.png?raw=true)

Comme je l'ai schématisé sur cette image, pour un noyau de taille 2 on a une profondeur de 6, pour 3 de 3 et pour 4 de 2, je n'ai pas représenté le noyau de taille 7 qui ne nécessite qu'une couche de CNN.

Entre chaque couche de CNN, la non-linéarité est une sigmoïde.

Après les couches de CNN, se trouvent des couches de RNN à taille variable, qui prend en entrée des vecteurs de 24 données et qui ressort des vecteurs de la même taille.

## Choix des hyper paramètres

Les hyper-paramètres que j'ai étudié sont les tailles des couches de CNN et de RNN ainsi que le learning rate et le nombre d'époques.

Dans un premier temps, sans avoir modifié aucun paramètre, les résultats sont assez bons, cela est du au grand nombre de données pour cette tâche. 
![essai1](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/essai1.png?raw=true)

On peut voir que les loss convergent vers 0.001 en quelques dizaines d'époques.


### Taille des CNN

Pour le nombre de couches des CNN, j'avais le choix entre les 4 taille de noyau/nombre de couches présentés au dessus.

![CNN_validation](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/CNN_validation.png?raw=true)

On peut voir que les performances des noyau de taille 2 et 7 sont moins bonnes sur le corpus de validation que celles des 3 et 4.

![cnntrain34](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/cnntrain34.png?raw=true)

En regardant ces courbes ont peut voir que sur l'ensemble train+validation l'architecture avec 3 couches se comporte mieux que celle à 2 couches (avec un etaille de noyau de 4).

### Taille des RNN

Les RNN prenant en entrée des vecteurs de la même taille que la sortie, il est facile de les empiler, cela est fait par la variable num_rnn du modèle.

![compa_rnn](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/compa_rnn.png?raw=true)

Après avoir regardé les résultats pour un nombre de couches variant de 1 à 9, les deux meilleurs présentés ici sont pour 2 et 8 couches, on préferera 2 couche pour ne pas avoir un trop grand nombre de paramètre.

### Learning rate

Après avoir fait varier le learning rate sur différents ordre de grandeurs (de 10^-5 à 10), un learning rate adapté semble être entre 0.01 et 0.001, pour que les loss soient stables et pas trop lentes.

![compa_lr](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/compa_lr.png?raw=true)

En vue de ce graphique, on choisit un learning rate de 0.005.

### Nombre d'époques

Avec les hyper-paramètres précédents, on utilise le jeu de test pour déterminer le bon nombre d'époques.

![epoch](https://github.com/alexiobec/deep_learning_mini_projet/blob/master/img/epoch.png?raw=true)

Les loss décroissent très vite, on pourrait choisir de s'arréter après 10 époques, cependant, la décroissance continue jusqu'à 20 époques, après elle réduit donc on peut choisir 20 époques.

#### Paramètres du modèle final :
  * CNN, taille des noyaux : 3, nombre de couche : 3
  * RNN, 2 couches
  * Learning rate = 0.005
  * Nombre d'époques = 20

## Conclusion

Ce modèle peret bien de prédire les exportations d'énergie françaises avec une grande précision.
