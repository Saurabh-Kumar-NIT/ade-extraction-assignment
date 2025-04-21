 Adverse Drug Event Extraction Using Generative NLP Models
This project explores the use of generative NLP models to extract Adverse Drug Events (ADEs) from medical forum posts using the CADEC dataset. It combines the power of LangChain, Hugging Face Transformers, and UMLS standardization APIs to build an agentic pipeline capable of not just extracting but also validating and correcting medical entities.

 Objective
To build an intelligent system that can:
Extract drugs, side effects (ADEs), and symptoms from patient-written texts
Standardize medical terms using UMLS
Automatically verify and improve its output using a generative retry mechanism

📚 Dataset: CADEC
The CADEC dataset contains real-world medical forum posts annotated with:
Drugs (mentioned by patients)
Adverse Drug Events (ADEs) (side effects)
Symptoms/Diseases (related conditions)
Each post captures the patient’s personal experience with medications, making it a valuable source for real-world ADE extraction.

📥 Dataset Link
⚙️ What This Project Does
1. Data Preprocessing
Loaded forum posts from the CADEC dataset
Cleaned and extracted relevant content
Used a Hugging Face generative model to expand medical abbreviations
Normalized and tokenized the text for further processing

2. Medical Entity Extraction
Used a generative model (like T5 or GPT-style models) to extract:
Drugs
Adverse Drug Events (ADEs)
Symptoms/Diseases
The model returns results in a clean JSON format

3. UMLS-Based Standardization
Queried the UMLS API to map extracted entities to standardized medical terms:
Drugs → RxNorm
ADEs & Symptoms → SNOMED CT
Replaced raw entity names with standardized terms using their Concept Unique Identifiers (CUIs)

4. Verification & Iterative Correction
Implemented multiple verification checks:
JSON format validation
Comparison with CADEC annotations
Semantic similarity using Sentence Transformers
If something failed, the system would automatically retry with feedback (up to 3 attempts), improving results over time

🔧 Tech Stack
Python
LangChain
Hugging Face Transformers
SentenceTransformers
UMLS API
Pandas, NumPy, JSON
