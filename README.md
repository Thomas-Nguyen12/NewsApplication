# NEWS APP
This is a Dash plotly dashboard app containing ML models for multilabel classification of news articles and the detection of AI-Generated news content

The main dashboard script is named "app.py" and can be run in the CLI using: 
```python3 app.py```

I have also created a separate streamlit dashboard solely for the ai-text detection model. This is found within the ai_detection_streamlit.py file and can be run using the command: 

```streamlit run ai_detection_streamlit.py```


### Objectives

1. Creating an ML model based on arvix data (listed below) to classify human-written and A.I-generated news articles.
2. Creating ML models to multilabel-classify news articles under news categories.


Note that: This project is **still in progress** and the documentation is not yet complete


### Manually sourced data

- For objective 1, Data extracted and processed manually is stored in the Data/ folder
- This data utilised a scrapy webscraper to extract content from Wikipedia under the url: https://en.wikipedia.org/wiki/Portal:Current_events/ from years 2004 to 2025 inclusive

### Data Sourced from Arvix
- For objective 2...
- Data extracted from "A Comprehensive Dataset for Human vs. AI Generated
Text Detection" (Roy et al., 2025) from Arvix  is stored within the arvix_data folder

## References

Roy, R., Imanpour, N., Aziz, A., Bajpai, S., Singh, G., Biswas, S., Wanaskar, K., Patwa, P., Ghosh, S., Dixit, S., Pal, N., Rawte, V., Garimella, R., Jena, G., Sheth, A., Sharma, V., Reganti, A., Jain, V., Chadha, A. and Das, A. (2025). A Comprehensive Dataset for Human vs. AI Generated Text Detection. [online] Arxiv.org. Available at: https://arxiv.org/html/2510.22874v1.
