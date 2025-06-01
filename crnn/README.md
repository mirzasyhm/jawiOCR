# Jawi OCR Project

This project provides scripts for Optical Character Recognition (OCR) on synthetic Jawi images using PyTorch. It includes utilities for downloading and handling the LMDB dataset, generating a character alphabet, and calculating Character Error Rate (CER).

## Project Structure

├── content/ # Directory for dataset (mimics Colab structure if download script is used)
│ └── data_jawi_color_2_lmdb/ # Extracted LMDB dataset (contains train, test, etc.)
├── download_dataset.py # Script to download and extract the LMDB dataset
├── build_alphabet.py # Script to generate the character alphabet from the dataset
├── dataset.py # PyTorch Dataset class for LMDB OCR data
├── utils.py # Utility functions (e.g., CER calculation)
├── requirements.txt # Python dependencies
├── alphabet.json # Generated alphabet (after running build_alphabet.py)
└── README.md # This file

