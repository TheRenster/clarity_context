# Clarity Context | News Analysis & Sentiment App
Clarity Context is a news analysis app that surfaces current events with balanced, context-driven framing — cutting through the negativity bias of traditional news without stripping out substance. Hosted on Streamlit.
What's in this repo (and why you might care)

streamlit_news_app.py – The main app: a Streamlit interface for browsing and exploring analyzed news stories with sentiment context.
news_analysis.py – Core analysis logic: fetches and processes news articles, runs sentiment scoring, and outputs structured results.
News_Analysis (2).ipynb – Exploratory notebook where the analysis pipeline was prototyped and refined.
news_analysis_results.csv – Output from the analysis pipeline: processed articles with sentiment scores and metadata.
sentiment_by_source.csv – Aggregated sentiment breakdown by news source, useful for comparing outlet-level tone.
cluster_analysis.csv – Results from topic/story clustering, grouping related articles together.
project_summary.csv – High-level summary data surfaced in the app dashboard.
requirements.txt – Python dependencies for running the app locally.
.devcontainer/ – Dev container config for a consistent development environment.

If you're interested in:

How to build a news aggregation and sentiment analysis pipeline in Python,
How to present NLP results in a clean, user-friendly Streamlit dashboard, or
The idea of consuming news more intentionally — with context instead of panic —

then this repo is worth exploring. If you're looking for a real-time, fully automated news pipeline at scale, this is a smaller, more focused starting point.
High-level structure

streamlit_news_app.py – Frontend app entry point
news_analysis.py – News fetching and sentiment analysis pipeline
News_Analysis (2).ipynb – Prototyping notebook
*.csv – Processed outputs: articles, sentiment, clusters, and summaries
requirements.txt – Dependencies
.devcontainer/ – Dev environment configuration
