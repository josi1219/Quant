from setuptools import setup, find_packages

setup(
    name="trading_model",
    version="0.1.0",
    description="Production EURUSD trading model — quant-level features, "
                "leak-proof training, cost-aware evaluation.",
    author="Yosef",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scipy>=1.10",
        "lightgbm>=4.0",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "tqdm>=4.65",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "matplotlib>=3.7", "seaborn>=0.12"],
        "export": ["onnx>=1.14", "onnxmltools>=1.12", "skl2onnx>=1.16"],
        "tune": ["optuna>=3.4"],
    },
)
