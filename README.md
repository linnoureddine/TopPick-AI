# TopPick AI 
**Bridging Technical Specs and Customer Reviews for Precise Laptop Recommendations**  

---

## 📌 Overview

TopPick AI is an AI-powered decision-making tool designed to assist users in selecting the best laptops based on technical specifications and sentiment analysis of real customer reviews. It combines advanced NLP, aspect-based sentiment analysis (ABSA), and a modified TOPSIS ranking algorithm to deliver personalized, trustworthy recommendations in minutes.

> 🧠 Unlike platforms that rely on biased or expert reviews, TopPick AI analyzes **authentic user reviews** and **real-world feedback**, making it ideal for both tech-savvy and non-technical users.

---

## 🌟 Features

- 💬 **Interactive Rasa Chatbot**  
  Collects user preferences and clarifies technical terms.

- 🔍 **Feature Extraction with InstructABSA**  
  Extracts laptop features (e.g., battery life, display) from reviews using a T5-based model.

- 😊 **Aspect-Based Sentiment Analysis with DeBERTa-v3**  
  Analyzes sentiment toward specific features mentioned in reviews.

- 🧲 **Sentiment-Enhanced TOPSIS Algorithm**  
  Ranks laptops based on weighted user preferences and adjusted sentiment scores.

- ⚖️ **Confidence Adjustment Mechanism**  
  Adjusts sentiment scores based on review count, helpfulness, and rating data.

- 🌟 **Bitmask Specification Filtering**  
  Matches laptops with user specifications using a binary-encoded filter.

---

## 💠 Tech Stack

| Component            | Technology Used                                           |
|---------------------|-----------------------------------------------------------|
| Chatbot             | [Rasa](https://rasa.com/)                                 |
| Feature Extraction  | T5-based `ate_tk-instruct-base-def-pos-laptops`           |
| Sentiment Analysis  | `yangheng/deberta-v3-base-absa-v1.1` (via PyABSA)         |
| Ranking Algorithm   | Modified TOPSIS with Confidence Factor                    |
| Frontend            | HTML/CSS/JS + SocketIO for integration                    |
| Backend             | Rasa Chatbot                                              |
| Dataset             | [Amazon Laptop Reviews (enriched)](https://huggingface.co/datasets/naga-jay/amazon-laptop-reviews-enriched) |
| NLP Libraries       | HuggingFace Transformers, PyTorch, spaCy                  |

---

## 🧱 System Architecture

```
User ➔ Rasa Chatbot ➔ User Specs Parser (Bitmask Filter) ➔
    ABSA Sentiment Pipeline
      └─ InstructABSA (T5)
      └─ DeBERTa-v3 ABSA
        ➔ Sentiment-Weighted TOPSIS Ranking
          ➔ Final Recommendation
```

---

## ⚙️ How It Works

1. **User Interaction**: The chatbot collects user specs and priorities.
2. **Filtering**: Bitmask-based spec filtering removes irrelevant laptops.
3. **Review Analysis**:
   - Extracts feature terms using InstructABSA.
   - Assigns sentiment using DeBERTa-v3.
   - Adjusts scores via confidence metrics (reviews, helpfulness, ratings).
4. **Ranking**: TOPSIS algorithm ranks laptops based on weighted feature sentiments.
5. **Result**: User receives the top laptops with explainable recommendations.

---

## 📂 Dataset

- Dataset: `amazon-laptop-reviews-enriched`
- Contains:
  - 2,612 unique laptops
  - 32,613 user reviews
  - Full product specs
- [Hugging Face Dataset Link](https://huggingface.co/datasets/naga-jay/amazon-laptop-reviews-enriched)

---

## 💻 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/TopPickAI.git
cd TopPickAI

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Rasa chatbot
cd chatbot
rasa train
rasa run

# In another terminal
rasa run actions

# Run the web app
cd ..
python app.py
```

---

## 🚀 Usage

1. Launch the chatbot and frontend.
2. Input your preferred laptop specs and prioritized features.
3. Get tailored laptop suggestions with explanations for each ranking.

---

## 📊 Evaluation

- **Accuracy**:
  - InstructABSA + DeBERTa-V3: **91%**
  - ChatGPT Pro (same dataset): **76%**

- **User Satisfaction**:
  - 80% of users preferred TopPick AI over ChatGPT in blind tests.

- **TOPSIS Personalization**:
  - Dynamic ranking based on weighted user priorities.

---

## Acknowledgements

- [Kevin Scaria](https://huggingface.co/kevinscaria) for T5-based InstructABSA.
- [Heng Yang](https://huggingface.co/yangheng) for DeBERTa-v3 ABSA model.
- HuggingFace, PyABSA, and the open-source NLP community.

---
