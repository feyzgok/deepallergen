DeepAllergen - Protein Allergen Prediction

DeepAllergen is a high-performance deep learning framework designed to predict protein allergenicity. By leveraging state-of-the-art ProtT5-XL embeddings and a hybrid CNN-Transformer architecture, it achieves superior accuracy in identifying potential allergens from primary amino acid sequences.


📁 Project Structure
```
deepallergen/
├── data/
│   ├── all.train.fasta
│   └──  all.test.fasta
│
├── src/
│   ├── data_preprocess.py
│   ├── train.py
│   ├── test.py
│   ├── graph_producer.py
│   ├── model.py
│   └── utils.py
│
├── preprocessed_data_t5/
├── saved_model/
├── output/
├── logs/
├── requirements.txt
├── run_pipeline.sh
└── README.md
```


Requirements
All dependencies are already installed in `.venv1`:
tensorflow==2.15.0
tensorflow-addons==0.23.0
tensorflow-metal==1.2.0 (Apple Silicon GPU support)
torch==2.8.0
transformers==4.57.0
scikit-learn==1.6.1
pandas==2.3.2
numpy==1.26.4
matplotlib==3.9.4


Installation
```bash
Navigate to project directory
cd ../PycharmProjects/deepallergen

 Virtual environment is already set up (.venv1)
 Activate it if not already active:
source .venv1/bin/activate  # Mac/Linux
or
.venv1\Scripts\activate  # Windows

Verify installation
python --version  # Should show Python 3.9.6
pip list | grep -E "tensorflow|torch|transformers"
```

📊 Data Format
FASTA Format
```
>allergen_protein1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVH
>non-allergen_protein2
MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVD
```

Supported Label Formats
- Allergen: `allergen`, `positive`, `1`, `pos`, `true`, `yes`
- Non-allergen: `protein`, `non-allergen`, `negative`, `0`, `neg`, `false`, `no`

🔧 Usage
Option 1: Automated Pipeline

```bash
Navigate to project
cd ../PycharmProjects/deepallergen

Ensure virtual environment is active
source .venv1/bin/activate

Make script executable (first time only)
chmod +x run_pipeline.sh

Run complete pipeline
./run_pipeline.sh
```

Option 2: Manual Step-by-Step

Step 1: Preprocess Data
```bash
cd ../PycharmProjects/deepallergen
source .venv1/bin/activate
python src/data_preprocess.py
```
Duration: 15-30 minutes

Output:
- `preprocessed_data/train_dataset`
- `preprocessed_data/val_dataset`
- `preprocessed_data/test_dataset`

Step 2: Train Model
```bash
python src/train.py
```
Training Duration:
- Apple Silicon (M1/M2/M3)**: 2-3 hours with Metal GPU
- Intel Mac (CPU only)**: 8-12 hours
- Linux/Windows GPU**: 2-4 hours

Output:
- `loss_grapb.jpg` - Training and validation loss curves
- Terminal statistics summary

Step 4: Test Model
```bash
python src/test.py
```
Output:
- `confusion_matrix.png`
- `prediction_analysis.png`
- Terminal metrics

📈 Output Interpretation
Training Metrics
- Loss: Binary cross-entropy (lower is better)
- Accuracy: Classification accuracy
- AUC: Area under ROC curve (>0.9 is excellent)

Test Results
🎯 TEST RESULTS
Accuracy: 0.95XX
Sensitivity (Recall): 0.94XX
Specificity: 0.96XX
Precision: 0.95XX
F1-Score: 0.95XX

Confusion Matrix:
True Positives (TP): XXX
True Negatives (TN): XXX
False Positives (FP): XX
False Negatives (FN): XX
```

Performance Benchmarks
- Accuracy > 0.90: Good
- Accuracy > 0.95: Excellent
- Sensitivity > 0.90: Good allergen detection
- Specificity > 0.90: Good non-allergen detection

📝 Model Architecture
DeepAllergen** combines:
- ProtT5-XL embeddings (1024-dim)
- Convolutional layers (kernel=11)
- Multi-head attention (8 heads)
- Feed-forward networks
- Dropout regularization

Input: Protein sequences (amino acids)
Output: Binary classification (allergen/non-allergen)

For questions or issues:
- Review `QUICKSTART.md` for common solutions
- Check logs in `output/` directory

📄 License
MIT License - see LICENSE file for details

Last Updated**: January 19, 2025
Version**: 1.0.0
