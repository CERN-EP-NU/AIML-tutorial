# AI/ML Tutorial Series for Beginners

This repository contains a comprehensive collection of Jupyter notebooks demonstrating various machine learning and deep learning models. Each notebook is self-contained with explanations, code, and visualizations.

## 📚 Contents

1. **XGBoost** (`01_xgboost.ipynb`) - Gradient boosting for classification and regression
2. **MLP** (`02_mlp.ipynb`) - Multi-Layer Perceptron neural networks
3. **CNN** (`03_cnn.ipynb`) - Convolutional Neural Networks for image processing
4. **GNN** (`04_gnn.ipynb`) - Graph Neural Networks for graph-structured data
5. **Autoencoder** (`05_autoencoder.ipynb`) - Autoencoders for dimensionality reduction
6. **VAE** (`06_vae.ipynb`) - Variational Autoencoders for generative modeling
7. **GAN** (`07_gan.ipynb`) - Generative Adversarial Networks
8. **Normalizing Flows** (`08_normalizing_flows.ipynb`) - Normalizing Flow models
9. **Transformer** (`09_transformer.ipynb`) - Attention-based models for sequences

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AIML-tutorial
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory and open any notebook to begin learning!

## 📊 Datasets

All notebooks use publicly available datasets that are automatically downloaded when you run the code. No manual dataset preparation is required.

## 🎯 Learning Path

For beginners, we recommend following this order:

1. Start with **01_xgboost.ipynb** to understand classification vs regression
2. Move to **02_mlp.ipynb** for your first neural network
3. Explore **03_cnn.ipynb** for image processing
4. Continue with other models based on your interests

## 📖 Key Concepts

### Classification vs Regression

- **Classification**: Predicting discrete categories (e.g., cat vs dog, spam vs not spam)
- **Regression**: Predicting continuous values (e.g., house prices, temperature)

Each applicable notebook demonstrates both tasks where relevant.

## 🛠️ Troubleshooting

If you encounter any issues:

1. Ensure your virtual environment is activated
2. Try upgrading pip: `pip install --upgrade pip`
3. Install dependencies one by one if bulk installation fails
4. Check that you have sufficient disk space for downloading datasets

## 📝 Requirements

See `requirements.txt` for a complete list of dependencies. Main libraries include:
- PyTorch (for deep learning)
- XGBoost (for gradient boosting)
- NumPy, Pandas (for data manipulation)
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for preprocessing and metrics)

## 🤝 Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements!

## 📄 License

This tutorial series is provided for educational purposes. Feel free to use and modify for your learning journey!