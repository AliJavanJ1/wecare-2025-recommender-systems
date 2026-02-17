[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FofgirAc)
# DataLabAssignement1

## generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those to you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt

## Implemented Methods (Matrix Completion / Collaborative Filtering)

In this project, we implemented and compared several families of recommenders for matrix completion:

- **Classical CF (SGD baseline):** global mean + user/item bias terms trained with SGD.
- **SGD + Metadata Features:** residual **Ridge regression** using item side features (genre, release year, popularity).
- **Bias-aware ALS:** alternating least squares matrix factorization with user/item biases.
- **Neural CF (NeuMF):** GMF + MLP architecture, enhanced with **SDAE side features** and stabilized via **K-fold CV + ensembling** (used for the final submission).
- **Deep Matrix Factorization (Deep-MF):** two MLP encoders for users/items with cosine similarity scoring + K-fold ensemble.
- **Graph Neural Networks (GNNs):** GCN and GAT variants on a heterogeneous user–movie–genre–decade graph.

## Ranking

🏆 Our team **wecare** ranked **#1** among all **M2 Master 2025** groups on the official leaderboard:  
https://www.lamsade.dauphine.fr/~testplatform/prds-a1/results/2025102202.html

## More Details

For full methodological details, experiments, and results, see **[report.pdf](https://github.com/AliJavanJ1/wecare-2025-recommender-systems/blob/main/report.pdf)** and **[slides.pdf](https://github.com/AliJavanJ1/wecare-2025-recommender-systems/blob/main/slides.pdf)**.