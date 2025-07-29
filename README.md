# VaxGuard: A Synthetic Dataset for Detecting Sources of Health Misinformation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VaxGuard is a synthetic benchmark designed to evaluate the ability of language models to detect and classify sources of health misinformation across multiple diseases. Unlike traditional misinformation datasets that focus on content veracity, VaxGuard emphasizes persona-driven misinformation by identifying who spreads misinformation and why.

---

## Key Features

- **Role‑Specific Classification**: Identifies four distinct misinformation roles  
- **Multi‑Disease Coverage**: Spans COVID‑19, HPV, and Influenza  
- **Cross‑Model Evaluation**: Tests generalizability across different LLMs  
- **Synthetic Data Generation**: Uses multiple LLMs (GPT‑4o, GPT‑3.5, Mistral, Phi‑3) for diverse narrative generation  

---

## Misinformation Roles

The dataset categorizes misinformation spreaders into four distinct roles:

| Role                          | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| **Religious Conspiracy Theorist** | Frames diseases as divine punishment or spiritual tests                   |
| **Anti‑Vaccine**              | Actively opposes vaccination based on pseudoscience or institutional distrust |
| **Fear Mongerer**             | Exaggerates risks and outcomes to incite panic                              |
| **Misinformation Spreader**   | Shares false information without clear ideological intent                   |

---

## Dataset Structure

\`\`\`
VaxGuard/
├── data/
│   ├── covid19/
│   ├── hpv/
│   └── influenza/
├── models/
│   ├── gpt_3.5/
│   ├── gpt_4o/
│   ├── mistral/
│   └── phi3/
└── evaluation/
    ├── cross_model_results/
    └── metrics/
\`\`\`

---

## Installation

\`\`\`bash
git clone https://github.com/your-username/vaxguard.git
cd vaxguard
pip install -r requirements.txt
\`\`\`

---

## Quick Start

### 1. Data Generation

Generate synthetic misinformation texts using different LLMs:

\`\`\`python
from vaxguard import DataGenerator

# Initialize generator
generator = DataGenerator(model="gpt-3.5-turbo")

# Generate samples for specific disease and role
samples = generator.generate_samples(
    disease="covid19",
    role="religious_conspiracy",
    num_samples=100
)
\`\`\`

### 2. Cross‑Model Evaluation

Evaluate model performance across different generation sources:

\`\`\`python
from vaxguard import CrossModelEvaluator

# Train on GPT-3.5 generated data, test on GPT-4o
evaluator = CrossModelEvaluator(
    train_model="gpt-3.5",
    test_model="gpt-4o"
)

results = evaluator.evaluate()
print(f"F1 Score: {results['f1']:.3f}")
\`\`\`

### 3. Role Classification

Classify misinformation roles in text:

\`\`\`python
from vaxguard import RoleClassifier

classifier = RoleClassifier.load_pretrained()
text = "This pandemic is a test sent by a higher power to punish humanity for its sins."
role = classifier.predict(text)
print(f"Predicted role: {role}")
\`\`\`

---

## Experimental Results

Our cross‑model evaluation reveals significant challenges in role classification:

| Model Pair           | Accuracy | Precision | Recall | F1 Score |
|----------------------|---------:|----------:|-------:|---------:|
| GPT‑3.5 → GPT‑4o     |     0.35 |      0.32 |   0.38 |     0.33 |
| GPT‑4o → Mistral     |     0.30 |      0.28 |   0.31 |     0.29 |
| Mistral → Phi‑3      |     0.38 |      0.36 |   0.39 |     0.37 |

**Key Findings:**

- No configuration exceeds 40% performance across metrics  
- Role semantics don’t consistently transfer across generation distributions  
- Cross‑disease generalization remains challenging  
- Models show sensitivity to prompt variations  

---

## File Structure

- \`GPT 3.5.py\` – GPT‑3.5 model implementation  
- \`GPT-4o.py\` – GPT‑4o model implementation  
- \`mistral.py\` – Mistral model implementation  
- \`VaxGuard_dataset.zip\` – Complete synthetic dataset  
 

---

## Methodology

### Data Generation Process

1. **Prompt Template Design**  
   Role‑specific templates for each disease.  
2. **LLM Generation**  
   Multiple models generate diverse narratives.  
3. **Quality Control**  
   Manual verification for semantic consistency.  
4. **Cross‑Model Setup**  
   Train/test splits across different generation models.

\`\`\`python
# Algorithm: Cross-Model Evaluation
for disease in ["covid19", "hpv", "influenza"]:
    for role in ["religious", "anti_vaccine", "fear_monger", "spreader"]:
        # Generate with model A, evaluate with model B
        train_data = generate_samples(model_a, disease, role)
        test_data = generate_samples(model_b, disease, role)
        classifier = train_classifier(train_data)
        results = evaluate_classifier(classifier, test_data)
\`\`\`

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository  
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)  
3. Commit your changes (\`git commit -m "Add amazing feature"\`)  
4. Push to the branch (\`git push origin feature/amazing-feature\`)  
5. Open a Pull Request  

---

## Citation

If you use VaxGuard in your research, please cite:

\`\`\`bibtex
@article{vaxguard2024,
  title={VaxGuard: A Synthetic Dataset for Detecting Sources of Health Misinformation},
  author={Anonymous ACL submission},
  journal={arXiv preprint},
  year={2024}
}
\`\`\`

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Ethics Statement

All generated data is synthetically constructed to avoid real‑world harm. We do not use or reproduce actual misinformation or user‑generated content. Experiments were conducted following ethical guidelines for responsible AI research.

---

## Limitations

- Relies on synthetic data which may not capture full real‑world complexity  
- Fixed role taxonomy may not generalize beyond our definitions  
- Limited to specific LLMs and may not represent all model behaviors  
- Performance metrics indicate current limitations in role classification accuracy  

---

## Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

⚠️ **Disclaimer:** This dataset is for research purposes only. The synthetic misinformation content should **not** be shared or used to spread actual misinformation.
