# Dimension Reduction

The following Python files and Jupyter notebooks serve to develop dimensionality reduction techniques for multi-lead ECG signals. Some examples of the strategies we employed were variational autoencoders, determenistic autoencoders, and PCA to reduce dimensionality from 100 --> 10. 

 * `multi_lead_autoencoder.py` trains a single autoencoder for a single patient across the target patients (is run on AWS), and saves encoded and reconstructed files
 * `normalize_heartbeats.py` calculates and creates histogram of the sentence similarity scores along off diagonals
 * `pca.py` investigates stemmed and lemmatized text and n-grams for example text
 * `reduction_error.py` computes the TF-IDF matrix and outputs the top-n words from the corpus of 187 text documents
 * `umap_viz.py` pairwise Universal Sentence Encoded (USE) sentence similarity (cosine) heatmap generated for a sample company
 * `vae.py` similar objective as above but includes t-SNE of visualizations of USE vectors

