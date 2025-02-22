Ge-SAND: An Explainable Deep Learning-Driven Framework for Disease Risk Prediction
Ge-SAND is a deep learning-based framework designed for predicting disease phenotypes by uncovering complex genetic interactions in parallel. This framework utilizes a series of models, including GESAN, FLBCN, and ELBCN, with a specialized GenomicEmbedding method to effectively model genetic interactions. The goal is to provide accurate disease risk predictions by leveraging genetic data and enhancing interpretability through explainable AI techniques.

File Overview
The repository includes the following files for model training, testing, and evaluation:

0_GESAN_FLBCN.py: Implements the GESAN + FLBCN model combination.
1_GESAN_ELBCN.py: Implements the GESAN + ELBCN model combination.
2_features_extraction.py: Handles feature extraction from genetic data.
3_GNDLN.py: Implements the Genomic Neural Dynamic Learning Network (GNDLN).
GeSAND.py: Main script for the Ge-SAND framework.
GenomicEmbedding.py: Defines the custom genomic embedding method used in the framework.
README.md: Documentation for using the repository and framework.
config.bin: Contains the hyperparameters used for model training. Adjust this file based on your specific environment or dataset.
gene_vocab.txt: Contains the vocabulary used for gene representation in the models.
requirements.txt: Lists the necessary dependencies to run the code, including libraries and their respective versions.
testing_GESAN_MEAN_BCN.py: A script for testing the output of GESAN combined with FLBCN or ELBCN models.
Environment Setup
To run the Ge-SAND framework, the following environment setup is required:

Install the dependencies listed in the requirements.txt file.
Ensure that you have the necessary data files (train, validate, test) ready for use.
Running the Framework
Once the necessary datasets are prepared, follow these steps in sequence:

Run 0_GESAN_FLBCN.py: This script trains the GESAN model combined with the FLBCN architecture.
Run 1_GESAN_ELBCN.py: This script trains the GESAN model combined with the ELBCN architecture.
Run 2_features_extraction.py: Extract features from the genetic data for further processing.
Run 3_GNDLN.py: Train the Genomic Neural Dynamic Learning Network (GNDLN) model.
After running these scripts, the model will be trained, and you can proceed to the evaluation phase.

Testing the Model Output
To test the outputs of the GESAN combined models (GESAN+FLBCN or GESAN+ELBCN), use the testing_GESAN_MEAN_BCN.py script.
