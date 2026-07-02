# RNA-seq Tissue-of-Origin Prediction for CUP

This repository contains the code developed for my MSc thesis project at the Technical University of Denmark (DTU) in collaboration with Rigshospitalet.

The project focuses on tissue-of-origin prediction for Cancer of Unknown Primary (CUP) using RNA-seq data. CUP is a heterogeneous group of metastatic cancers where the primary tumor site cannot be identified through standard diagnostic work-up. The main objective of this thesis is to evaluate whether transcriptomic profiles can be used to infer the likely tissue of origin and support future diagnostic decision-making.

The repository includes scripts for preprocessing RNA-seq data, training variational autoencoder-based models, generating latent embeddings, training multinomial logistic regression classifiers, evaluating internal and external performance, and visualizing the learned latent space using UMAP. The models are trained on reference transcriptomic datasets and evaluated on an independent metastatic cohort from Rigshospitalet.
