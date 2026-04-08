# News Shield

A machine learning pipeline for news analysis, combining AI-generated text detection, multi-label topic classification, and sentiment-driven stock price forecasting.

**Please Note:** The technical documentation for the Streamlit Dashboard - NewsShield_Documentation.pdf - is not complete and is still being updated.

**Live Demo:** [News Shield Dashboard](https://newsshield.streamlit.app)

```bash
streamlit run ai_detection_streamlit.py
```

---

## Overview

News Shield is a multi-objective research project that applies natural language processing and machine learning to three interconnected problems in news analysis:

| Objective | Description |
|-----------|-------------|
| **Topic Classification** | Multi-label classification of news articles into topic categories |
| **AI Detection** | Binary classification of human-written vs. AI-generated news articles |
| **Sentiment Forecasting** | Stock price time-series analysis augmented with sentiment features |

---

## Objectives

### Objective 1 — News Topic Classification (Multi-Label)

Trains a multi-label classifier to assign news articles to one or more topic categories.

- **Data source:** Wikipedia Current Events Portal (`https://en.wikipedia.org/wiki/Portal:Current_events/`), scraped across years 2004–2025 using a custom Scrapy web scraper
- **Storage:** Processed data is stored in the `data/` directory
- **Model artefacts:** `models/news_topic_classifier/`

---

### Objective 2 — AI-Generated News Detection

Trains a binary classifier to distinguish human-written news articles from AI-generated ones, with the goal of flagging potentially skewed or synthetic content.

- **Data source:** *A Comprehensive Dataset for Human vs. AI Generated Text Detection* (Roy et al., 2025)
- **Storage:** Raw and processed data is stored in the `arvix_data/` directory
- **Model artefacts:** `models/`

---

### Objective 3 — Stock Price Forecasting with Sentiment Analysis

Extends a time-series forecasting model for stock prices by incorporating financial sentiment scores as an auxiliary variable.

- **Data source:** Financial phrasebank from *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts* (Malo et al., 2013)
- **Storage:** Data is stored in the `sentiment_analysis_data/` directory

---

## Project Structure

```
news_project/
├── arvix_data/                          # AI detection datasets
│   ├── arvix_dataset.csv
│   ├── data_formatted.csv
│   └── X.csv
├── data/                                # Topic classification datasets
│   ├── raw/
│   │   └── scraped_news_2004_2025.csv
│   ├── clean_df.csv
│   ├── cleaned_df.csv
│   └── news_data_ready_to_plot.csv
├── models/
│   ├── ai_detector/                     # AI detection model artefacts
│   │   ├── ai_detector.pkl
│   │   └── ai_vectoriser.pkl
│   ├── news_topic_classifier/           # Topic classifier artefacts
│   │   ├── mlb.pkl
│   │   ├── news_topic_classifier.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── sentiment_analyser/             # Sentiment model artefacts
│       ├── sentiment_analyser.pkl
│       └── sentiment_vectoriser.pkl
├── news_scraper/                        # Scrapy web scraper
│   ├── news_scraper/
│   │   ├── spiders/
│   │   │   └── scraper.py
│   │   ├── items.py
│   │   ├── middlewares.py
│   │   ├── pipelines.py
│   │   └── settings.py
│   └── scrapy.cfg
├── notebooks/                           # Exploratory analysis notebooks
│   ├── analyse_clean_df.ipynb
│   ├── fake_news.ipynb
│   ├── improve_model.ipynb
│   └── sentiment_analysis.ipynb
├── sentiment_analysis_data/             # Sentiment & stock forecasting data
│   └── sentiment_data.csv
├── ai_detection.py                      # AI detection pipeline
├── ai_detection_streamlit.py            # Streamlit dashboard
├── classify_news.py                     # Topic classification pipeline
├── sentiment_analyser.py                # Sentiment analysis pipeline
├── NewsShield_Documentation.docx        # Project documentation
├── NewsShield_Documentation.pdf
├── ny_times.jpg
├── requirements.txt
└── README.md
```

---

## References

Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2013). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1307.5336

Roy, R., Imanpour, N., Aziz, A., Bajpai, S., Singh, G., Biswas, S., Wanaskar, K., Patwa, P., Ghosh, S., Dixit, S., Pal, N., Rawte, V., Garimella, R., Jena, G., Sheth, A., Sharma, V., Reganti, A., Jain, V., Chadha, A. and Das, A. (2025). A Comprehensive Dataset for Human vs. AI Generated Text Detection. *arXiv*. https://arxiv.org/html/2510.22874v1
