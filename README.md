# Autobin

Combining multiple biomarker information and clinical information can offer a more complete and comprehensive information to predict treatment response and reveal the underlying disease mechanisms than using single biomarker or clinical data alone. To address this, we developed Autobin, a tool that integrates various biomarker and clinical data to enhance predictive modeling in biomedical research.




## Model Structure

Autobin consists of three key components:
- KL-PMVAE: A pathway-masked variational autoencoder for biologically informed data compression and latent feature extraction.
- LFbin: A latent-feature-fed multi-layer perceptron for responder prediction.
- DeepSHAP: An integrated DeepSHAP interpretation module for feature attribution and model explainability.

For simplicity, all components are integrated into a single Python script, `train.py`, which provides a command line interface. The script first trains the KL-PMVAE model and determines the optimal latent dimension using cross-validation. Next, it uses the extracted latent features along with clinical biomarkers to train the LFbin model, applying 5-fold cross-validation to assess overall performance. Finally, DeepSHAP is applied to the best LFbin and PMVAE models to generate feature attributions and explainability results.

## User Guide

To run Autobin, you have two options:

1. **On Magellan**:  
    A Conda environment named `autobin` is already installed. Simply activate it and run the training script:
    ```bash
    conda activate autobin
    python train.py
    ```

2. **On your own machine**:  
    First, create the Conda environment using the provided `environment.yml` file, then activate it and run the training script:
    ```bash
    conda env create -f environment.yml
    conda activate autobin
    python train.py
    ```



### Command Line Arguments

To increase model flexibility and usability, Autobin's `train.py` script accepts several command line arguments. These allow you to specify paths for input data files and model output locations. The default paths are set to common directories, but you can customize them as needed.

- `--clinical_data`: Path to the clinical data CSV file.  
    *Default*: `data/clinical_data.csv`
- `--gene_data`: Path to the gene expression data CSV file.  
    *Default*: `data/gene_data.csv`
- `--other_data`: Path to other data (omics) CSV file.  
    *Default*: `data/other_data.csv`
- `--pathway_data`: Path to the pathway mask CSV file.  
    *Default*: `data/pathway.csv`
- `--response_data`: Path to the response variable CSV file.  
    *Default*: `data/response_data.csv`
- `--latent_dim_data`: Path to save the latent dimension data CSV file.  
    *Default*: `data/latent_dim.csv`
- `--pmvae_path`: Path to save or load the PMVAE model.  
    *Default*: `model/pmvae_model.pt`
- `--lfbin_path`: Path to save or load the LFBin model directory.  
    *Default*: `model/lfbin`

    > **⚠️ Attention:**  
    > Ensure your input data files are properly formatted (same order across files) and preprocessed on the same scaling before running `train.py`. Incorrect or missing data may cause errors or unexpected results.



## Contact

For questions or support, contact [Yifan](mailto:ywu3939@gmail.com) or [Weiliang](mailto:Weiliang.Qiu@sanofi.com).