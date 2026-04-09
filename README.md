# News Shield

A machine learning pipeline for news analysis, combining AI-generated text detection, multi-label topic classification, and sentiment-driven stock price forecasting.

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
.
├── arvix_data          # data used for the models/ai_detector (objective 2)
├── data                # scraped news data used for the multilabel news topic classifier (objective 1) 
│   └── raw
├── documentation
├── images
├── models
│   ├── ai_detector         
│   ├── news_topic_classifier
│   └── sentiment_analyser
├── news_scraper        # scrapy web scraper used to collect data for news topic classifier (objective 1)
│   └── news_scraper
│       └── spiders
├── notebooks
├── scripts             # wrapper scripts used to implement the machine learning models within models/ 
└── sentiment_analysis_data         # data collected from Malo et al. (2013) to analyse sentiment within financial news articles
```

---

**Please Note:** 

1. The technical documentation for the Streamlit Dashboard - NewsShield_Documentation.pdf - is not complete and is still being updated.

2. News Shield currently only supports Objectives 1 & 2, with objective 3 still under development


## References

Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2013). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1307.5336

Roy, R., Imanpour, N., Aziz, A., Bajpai, S., Singh, G., Biswas, S., Wanaskar, K., Patwa, P., Ghosh, S., Dixit, S., Pal, N., Rawte, V., Garimella, R., Jena, G., Sheth, A., Sharma, V., Reganti, A., Jain, V., Chadha, A. and Das, A. (2025). A Comprehensive Dataset for Human vs. AI Generated Text Detection. *arXiv*. https://arxiv.org/html/2510.22874v1
