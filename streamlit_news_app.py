"""
News Feed Dashboard - Stay Informed Without the Overwhelm
A Streamlit app for exploring news with sentiment analysis, clustering, and summarization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
# Add these to your imports at the top
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from newsapi import NewsApiClient

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data once"""
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return True

download_nltk_data()

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Page configuration
st.set_page_config(
    page_title="Clarity Context",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Main background - match sidebar blue */
    .main {
        background-color: #0e1117;
    }
    
    /* Main content area */
    .block-container {
        background-color: #0e1117;
        padding-top: 2rem;
    }
    
    /* Sidebar styling with blue theme */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
            
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stTextInput input {
        background-color: white;
        color: #333;
        border: 2px solid #3498db;
    }
    
    [data-testid="stSidebar"] .stMultiSelect {
        color: #333;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white !important;
    }
    
    /* Sidebar multiselect boxes - WHITE background */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: white !important;
    }
    
    /* Multiselect dropdown container - white */
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: white !important;
        border-radius: 8px;
    }
    
    /* Individual selected tags - SUPER LIGHT GRAY */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #ffffff !important;
        color: #333 !important;
    }
    
    /* Filter containers - white background */
    [data-testid="stSidebar"] > div {
        background-color: #5dade2;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    .article-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
    
    .sentiment-neutral {
        background-color: #f8f9fa;
        color: #6c757d;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
    
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
    }
    
    .tag {
        background-color: #e9ecef;
        color: #495057;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 12px;
        margin-right: 5px;
        display: inline-block;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #3498db 0%, #5dade2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e1117;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #3498db;
        color: white;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 10px 18px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #3498db;
    }
    
    /* Info boxes - light background */
    .stAlert {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function
@st.cache_data(ttl=21600)  # Cache for 6 hours
def fetch_and_analyze_news(api_key):
    """
    Fetch news from API and perform all analysis
    Returns: analyzed DataFrame
    """
    
    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=api_key)
    
    # Define sources with metadata
    sources = {
        "reuters": {"leaning": "Center", "country": "UK"},
        "associated-press": {"leaning": "Center", "country": "USA"},
        "bbc-news": {"leaning": "Center", "country": "UK"},
        "al-jazeera-english": {"leaning": "Center-Left", "country": "Qatar"},
        "cnn": {"leaning": "Left-Center", "country": "USA"},
        "fox-news": {"leaning": "Right", "country": "USA"},
        "the-wall-street-journal": {"leaning": "Right-Center", "country": "USA"},
        "bloomberg": {"leaning": "Center", "country": "USA"},
    }
    
    # Fetch articles
    all_articles = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (source, info) in enumerate(sources.items()):
        try:
            status_text.text(f"Fetching from {source}...")
            response = newsapi.get_everything(
                sources=source,
                language='en',
                page_size=50,
                sort_by='publishedAt'
            )
            
            for article in response['articles']:
                all_articles.append({
                    'source': source,
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'publishedAt': article['publishedAt'],
                    'url': article['url'],
                    'leaning': info['leaning'],
                    'country_of_origin': info['country']
                })
            
            progress_bar.progress((idx + 1) / len(sources))
            
        except Exception as e:
            st.warning(f"Error fetching {source}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    
    # Clean data
    df = df.drop_duplicates(subset=['title'])
    df = df.dropna(subset=['content', 'description'])
    
    # Text cleaning
    df['clean_text'] = df['content'].apply(clean_text)
    df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['clean_full_text'] = df['full_text'].apply(clean_text)
    df = df[df['clean_full_text'].str.len() >= 10]
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        max_features=1000
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_full_text'])
    
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if pd.isna(text) or text == "":
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
        return sia.polarity_scores(text)
    
    df['sentiment_scores'] = df['full_text'].apply(get_sentiment)
    df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    
    def sentiment_label(compound):
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment_label'] = df['sentiment_compound'].apply(sentiment_label)
    
    # K-Means Clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Fix political leaning labels
    leaning_map = {
        "Left": "Left",
        "Left-Center": "Left-Center",
        "Center-Left": "Left-Center",
        "Center": "Center",
        "Right-Center": "Right-Center",
        "Right": "Right"
    }
    df['leaning'] = df['leaning'].map(leaning_map).fillna(df['leaning'])
    
    # Add empty summary column
    df['summary'] = ''
    
    return df

# Load data
# API Key configuration
st.sidebar.header("Configuration")

# Check for API key in secrets (Streamlit Cloud) or use default
try:
    if 'newsapi_key' in st.secrets:
        api_key = st.secrets['newsapi_key']
    else:
        api_key = st.sidebar.text_input("NewsAPI Key", type="password", value="145fafc1f2264a0f91d8e355a6f0b2c5")
except:
    # No secrets file exists (local development)
    api_key = st.sidebar.text_input("NewsAPI Key", type="password", value="145fafc1f2264a0f91d8e355a6f0b2c5")

# Refresh button
if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# Load data
if api_key:
    try:
        with st.spinner("Fetching and analyzing news articles..."):
            df = fetch_and_analyze_news(api_key)
        
        st.sidebar.success(f"Loaded {len(df)} articles")
        st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    st.warning("Please enter your NewsAPI key")
    st.stop()
    
# App title with your custom logo
try:
    st.image('clarity_context_logo.png', use_container_width=True)
    st.markdown(
        """
        <div style="text-align:center;">
            <p style="color: #666; font-size: 18px; margin-top: 10px;">
                Stay informed without the overwhelm - news filtered, summarized, and organized
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    # Fallback if image not found
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-family: 'Inter', sans-serif; font-size: 56px; font-weight: 700; color: #333; margin: 0;">
                Clarity <span style="color: #3498db;">C</span>ontext
            </h1>
            <p style="color: #666; font-size: 18px; margin-top: 10px;">
                Stay informed without the overwhelm - news filtered, summarized, and organized
            </p>
        </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["DASHBOARD", "STORY STATS", "TEXT STATS"])

# =====================================================================
# TAB 1: DASHBOARD (Main News Feed)
# =====================================================================
with tab1:
    # Sidebar filters
    st.sidebar.header("SEARCH & FILTER")
    
    # Search box
    search_term = st.sidebar.text_input("Search articles", placeholder="Type keywords...")
    
    # Genre/Topic filter (using clusters as topics)
    st.sidebar.subheader("Topic Categories")
    
    # Map clusters to topic names
    cluster_names = {
        0: "Politics",
        1: "Business", 
        2: "International",
        3: "Technology",
        4: "General News"
    }
    
    # Create readable topic options
    all_clusters = sorted(df['cluster'].unique())
    topic_options = st.sidebar.multiselect(
        "Filter by topic",
        options=all_clusters,
        default=all_clusters,
        format_func=lambda x: cluster_names.get(x, f'Topic {x}')
    )
    
    # Sentiment filter
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=['Positive', 'Neutral', 'Negative'],
        default=['Positive', 'Neutral', 'Negative']
    )
    
    # Source filter
    source_filter = st.sidebar.multiselect(
        "News Source",
        options=sorted(df['source'].unique()),
        default=sorted(df['source'].unique())
    )
    
    # Political leaning filter
    leaning_filter = st.sidebar.multiselect(
        "Political Leaning",
        options=sorted(df['leaning'].unique()),
        default=sorted(df['leaning'].unique())
    )
    
    # Apply filters
    filtered_df = df[
        (df['cluster'].isin(topic_options)) &
        (df['sentiment_label'].isin(sentiment_filter)) &
        (df['source'].isin(source_filter)) &
        (df['leaning'].isin(leaning_filter))
    ]
    
    # Apply search
    if search_term:
        mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) |
            filtered_df['description'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Sort options
    st.sidebar.subheader("Sort By")
    sort_by = st.sidebar.radio(
        "Order articles by",
        options=['Most Recent', 'Most Positive', 'Most Negative', 'Source'],
        index=0
    )
    
    if sort_by == 'Most Recent':
        filtered_df = filtered_df.sort_values('publishedAt', ascending=False)
    elif sort_by == 'Most Positive':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=False)
    elif sort_by == 'Most Negative':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=True)
    else:
        filtered_df = filtered_df.sort_values('source')
    
    # Main content area
    st.markdown("---")
    
    # Quick stats at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", len(filtered_df))
    with col2:
        avg_sent = filtered_df['sentiment_compound'].mean()
        st.metric("Avg Sentiment", f"{avg_sent:.2f}", 
                    help="Average sentiment score across all articles. Range: -1 (very negative) to +1 (very positive)")
    with col3:
        pos_pct = (filtered_df['sentiment_label'] == 'Positive').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Positive %", f"{pos_pct:.0f}%",
                    help="Percentage of articles classified as positive (sentiment > 0.05)")
    with col4:
        st.metric("Topics", filtered_df['cluster'].nunique(),
                    help="Number of different topic clusters in filtered results")
    
    st.markdown("---")
    
    # Display articles in feed format
    st.subheader(f"Latest Stories ({len(filtered_df)} articles)")
    
    if len(filtered_df) == 0:
        st.warning("Sorry... looks like we don't have any stories on that currently, please search for another topic and check back on this one later!")
    else:
        # Pagination
        articles_per_page = 20
        total_pages = max(1, (len(filtered_df) - 1) // articles_per_page + 1)
        
        # Initialize session state for page number if not exists
        if 'page_num' not in st.session_state:
            st.session_state.page_num = 1
        
        # Reset to page 1 if out of bounds
        if st.session_state.page_num > total_pages:
            st.session_state.page_num = 1
        
        start_idx = (st.session_state.page_num - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, len(filtered_df))
        
        for idx, article in filtered_df.iloc[start_idx:end_idx].iterrows():
            # Calculate text stats for this article - using ORIGINAL content only
            # content_text = str(article.get('content', ''))
            
            # word_count = len(content_text.split())
            # char_count = len(content_text)
            # sentence_count = len([s for s in content_text.split('.') if s.strip()])
            # avg_word_length = sum(len(word) for word in content_text.split()) / max(word_count, 1)
            
            # Create article card
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Article title with date
                    # pub_date = article['publishedAt'].strftime('%b %d, %Y')
                    pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    
                    # # Title with hover stats - NO EMOJI
                    # with st.popover("Text Stats", use_container_width=False):
                    #     st.markdown("**Article Text Statistics:**")
                    #     st.metric("Word Count", f"{word_count:,}")
                    #     st.metric("Character Count", f"{char_count:,}")
                    #     st.metric("Sentences", sentence_count)
                    #     st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")
                    #     if word_count > 0:
                    #         st.metric("Reading Time", f"{word_count // 200} min")
                    
                    st.markdown(f"### {article['title']}")
                    st.caption(f"{pub_date}")
                    
                    # Color-coded tags
                    # Cluster tag (topic-based colors)
                    cluster_colors = {
                        0: "#3498db",  # Blue for Politics
                        1: "#2ecc71",  # Green for Business
                        2: "#9b59b6",  # Purple for International
                        3: "#e67e22",  # Orange for Technology
                        4: "#95a5a6"   # Gray for General
                    }
                    cluster_color = cluster_colors.get(article['cluster'], "#95a5a6")
                    
                    # Leaning tag (political spectrum colors)
                    leaning_colors = {
                        "Left": "#1e88e5",
                        "Left-Center": "#42a5f5",
                        "Center-Left": "#64b5f6",
                        "Center": "#9e9e9e",
                        "Right-Center": "#ef5350",
                        "Right": "#d32f2f"
                    }
                    leaning_color = leaning_colors.get(article['leaning'], "#9e9e9e")
                    
                    # Country tag (region-based colors)
                    country_colors = {
                        "USA": "#e74c3c",
                        "UK": "#3498db",
                        "Qatar": "#16a085"
                    }
                    country_color = country_colors.get(article['country_of_origin'], "#95a5a6")
                    
                    tag_html = f"""
                    <div style="margin: 10px 0;">
                        <span style="background-color: {cluster_color}; color: white; padding: 3px 8px; border-radius: 5px; font-size: 12px; margin-right: 5px;">
                            {cluster_names.get(article['cluster'], f"Topic {article['cluster']}")}
                        </span>
                        <span style="background-color: #6c757d; color: white; padding: 3px 8px; border-radius: 5px; font-size: 12px; margin-right: 5px;">
                            {article['source']}
                        </span>
                        <span style="background-color: {leaning_color}; color: white; padding: 3px 8px; border-radius: 5px; font-size: 12px; margin-right: 5px;">
                            {article['leaning']}
                        </span>
                        <span style="background-color: {country_color}; color: white; padding: 3px 8px; border-radius: 5px; font-size: 12px; margin-right: 5px;">
                            {article['country_of_origin']}
                        </span>
                    </div>
                    """
                    st.markdown(tag_html, unsafe_allow_html=True)
                    
                    # Summary
                    if 'summary' in article and pd.notna(article['summary']) and article['summary'] != '':
                        st.markdown(f"**Summary:** {article['summary']}")
                    elif 'description' in article and pd.notna(article['description']):
                        st.markdown(f"**Description:** {article['description']}")
                    
                    # Original link as button
                    st.markdown(f"[Read Full Article]({article['url']})", unsafe_allow_html=True)
                
                with col2:
                    # Sentiment badge
                    sentiment = article['sentiment_label']
                    score = article['sentiment_compound']
                    
                    if sentiment == 'Positive':
                        st.markdown(f'<div class="sentiment-positive">POSITIVE<br/>{score:.2f}</div>', unsafe_allow_html=True)
                    elif sentiment == 'Negative':
                        st.markdown(f'<div class="sentiment-negative">NEGATIVE<br/>{score:.2f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="sentiment-neutral">NEUTRAL<br/>{score:.2f}</div>', unsafe_allow_html=True)
                    
                    # Sentiment breakdown with explanation
                    st.caption("Sentiment Breakdown:")
                    st.caption(f"Positive: {article['sentiment_pos']:.2f}")
                    st.caption(f"Negative: {article['sentiment_neg']:.2f}")
                    st.caption(f"Neutral: {article['sentiment_neu']:.2f}")
                    st.caption("(These 3 add up to 1.0)")
                    
                    # Calculate sensationalism score
                    # Higher absolute sentiment + lower neutral = more sensational
                    sensational_score = abs(article['sentiment_compound']) * (1 - article['sentiment_neu'])
                    
                    st.caption("---")
                    # Sensationalism with hover info icon matching the top metrics
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.caption("Sensationalism Level:")
                    with col_b:
                        st.caption("")  # spacer
                    
                    # Add help text that appears on hover
                    sensational_help = """
Sensationalism Score = |compound| × (1 - neutral)

• HIGH (>0.3): Very emotional language, strong opinions
• MEDIUM (0.15-0.3): Moderate emotional content  
• LOW (<0.15): Factual, neutral reporting

Articles with extreme sentiment and little neutral language score higher.
                    """
                    
                    if sensational_score > 0.3:
                        st.markdown("**HIGH**", unsafe_allow_html=True, help=sensational_help)
                        st.caption(f"Score: {sensational_score:.2f}")
                    elif sensational_score > 0.15:
                        st.markdown("**MEDIUM**", unsafe_allow_html=True, help=sensational_help)
                        st.caption(f"Score: {sensational_score:.2f}")
                    else:
                        st.markdown("**LOW**", unsafe_allow_html=True, help=sensational_help)
                        st.caption(f"Score: {sensational_score:.2f}")
                
                st.markdown("---")
        
        # Navigation buttons and info ONLY at bottom
        st.markdown("---")
        st.info(f"Showing articles {start_idx + 1}-{end_idx} of {len(filtered_df)} total")
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col2:
            if st.button("← Previous", disabled=(st.session_state.page_num == 1), use_container_width=True, key="prev_bottom"):
                st.session_state.page_num -= 1
                st.rerun()
        
        with col3:
            st.markdown(f"<h4 style='text-align: center;'>Page {st.session_state.page_num} of {total_pages}</h4>", unsafe_allow_html=True)
        
        with col4:
            if st.button("Next →", disabled=(st.session_state.page_num == total_pages), use_container_width=True, key="next_bottom"):
                st.session_state.page_num += 1
                st.rerun()

# =====================================================================
# TAB 2: STORY STATS (Visualizations)
# =====================================================================
with tab2:
    st.header("Story Statistics & Analysis")
    
    # Important note about sentiment
    with st.expander("ℹ️ Understanding Sentiment Scores", expanded=False):
        st.markdown("""
        ### How Sentiment Scoring Works
        
        **VADER Sentiment Analysis** provides multiple scores:
        
        1. **Pos, Neg, Neu** (always add up to 1.0)
            - These show the *proportion* of positive, negative, and neutral language
            - Example: pos=0.91, neg=0.09, neu=0.00 means 91% positive words, 9% negative
        
        2. **Compound Score** (-1 to +1)
            - This is the *overall* sentiment considering context and intensity
            - It's NOT just (pos - neg)
            - Uses linguistic rules to determine if the overall message is positive or negative
            - Example: "The movie was not bad" → High neg proportion, but compound is positive!
        
        **Why compound can be negative when pos > neg:**
        - Negations ("not good", "barely acceptable")
        - Sarcasm or mixed sentiment
        - Negative context outweighs positive words
        
        **Note on Political Bias:**
        - Sentiment scores measure *emotional tone*, not *political favorability*
        - CNN might have positive sentiment about Trump events if the article is optimistic
        - Fox might have negative sentiment if discussing Democrat criticisms
        - These scores reflect article *tone*, not source *bias*
        """)
    
    # Use filtered data from main dashboard
    st.info(f"Analyzing {len(filtered_df)} articles based on your current filters")
    
    # Sentiment analysis section
    st.subheader("Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#95a5a6',
                'Negative': '#e74c3c'
            },
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by source
        sentiment_by_source = filtered_df.groupby('source')['sentiment_compound'].mean().sort_values()
        fig = px.bar(
            x=sentiment_by_source.values,
            y=sentiment_by_source.index,
            orientation='h',
            title='Average Sentiment by Source',
            labels={'x': 'Average Sentiment Score', 'y': 'Source'},
            color=sentiment_by_source.values,
            color_continuous_scale='RdYlGn'
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Topic clustering section
    st.subheader("Topic Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster distribution - sorted by count
        cluster_counts = filtered_df['cluster'].value_counts().sort_values(ascending=False)
        cluster_labels = [cluster_names.get(i, f"Topic {i}") for i in cluster_counts.index]
        
        fig = px.bar(
            x=cluster_labels,
            y=cluster_counts.values,
            title='Articles per Topic Cluster',
            labels={'x': 'Topic', 'y': 'Number of Articles'},
            color=cluster_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by cluster
        cluster_sentiment = filtered_df.groupby('cluster')['sentiment_compound'].mean().sort_values()
        cluster_labels_sent = [cluster_names.get(i, f"Topic {i}") for i in cluster_sentiment.index]
        
        fig = px.bar(
            x=cluster_labels_sent,
            y=cluster_sentiment.values,
            title='Average Sentiment by Topic',
            labels={'x': 'Topic', 'y': 'Average Sentiment'},
            color=cluster_sentiment.values,
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Political leaning analysis
    st.subheader("Political Leaning Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Articles by leaning
        leaning_counts = filtered_df['leaning'].value_counts()
        fig = px.bar(
            x=leaning_counts.index,
            y=leaning_counts.values,
            title='Articles by Political Leaning',
            labels={'x': 'Political Leaning', 'y': 'Number of Articles'},
            color=leaning_counts.values,
            color_continuous_scale='Purples'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by leaning (boxplot)
        fig = px.box(
            filtered_df,
            x='leaning',
            y='sentiment_compound',
            title='Sentiment Distribution by Political Leaning',
            labels={'leaning': 'Political Leaning', 'sentiment_compound': 'Sentiment Score'},
            color='leaning'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Timeline
    st.subheader("Publication Timeline")
    filtered_df['publishedAt'] = pd.to_datetime(filtered_df['publishedAt'], errors='coerce')
    timeline_df = (
    filtered_df
        .dropna(subset=['publishedAt'])
        .groupby(filtered_df['publishedAt'].dt.date)
        .size()
        .reset_index()
    )

    timeline_df.columns = ['Date', 'Count']

    
    fig = px.area(
        timeline_df,
        x='Date',
        y='Count',
        title='Articles Published Over Time',
        labels={'Count': 'Number of Articles'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 3: TEXT STATISTICS
# =====================================================================
with tab3:
    st.header("Text Statistics & Word Analysis")
    
    # Use filtered data
    st.info(f"Analyzing {len(filtered_df)} articles based on your current filters")
    
    # Calculate overall text stats
    st.subheader("Overall Text Statistics")
    
    # Add word count column if not exists
    if 'word_count' not in filtered_df.columns:
        def calculate_word_count(row):
            content = str(row.get('content', '') or '')
            description = str(row.get('description', '') or '')
            full_text = content + ' ' + description
            return len(full_text.split())
        
        filtered_df['word_count'] = filtered_df.apply(calculate_word_count, axis=1)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_words = filtered_df['word_count'].sum()
        st.metric("Total Words", f"{total_words:,}")
    
    with col2:
        avg_words = filtered_df['word_count'].mean()
        st.metric("Avg Words/Article", f"{avg_words:.0f}")
    
    with col3:
        max_words = filtered_df['word_count'].max()
        st.metric("Longest Article", f"{max_words:,} words")
    
    with col4:
        min_words = filtered_df['word_count'].min()
        st.metric("Shortest Article", f"{min_words:,} words")
    
    st.markdown("---")
    
    # Words per source (sorted most to least)
    st.subheader("Total Words by Source")
    
    words_by_source = filtered_df.groupby('source')['word_count'].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=words_by_source.index,
        y=words_by_source.values,
        title='Total Word Count by News Source (Most to Least)',
        labels={'x': 'Source', 'y': 'Total Words'},
        color=words_by_source.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Average words per article by source
    st.subheader("Average Article Length by Source")
    
    avg_words_by_source = filtered_df.groupby('source')['word_count'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=avg_words_by_source.index,
        y=avg_words_by_source.values,
        title='Average Words per Article by Source',
        labels={'x': 'Source', 'y': 'Average Words'},
        color=avg_words_by_source.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top words analysis
    st.subheader("Most Frequent Words Across All Articles")
    
    # Combine all text
    from collections import Counter
    import re
    
    all_text = ' '.join(
        filtered_df['clean_text'].fillna('').astype(str).tolist()
    ).lower()
    
    # Remove common words and get top 20
    words = all_text.split()
    word_counts = Counter(words)
    
    # Get top 20
    top_20_words = dict(word_counts.most_common(20))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig = px.bar(
            x=list(top_20_words.keys()),
            y=list(top_20_words.values()),
            title='Top 20 Most Frequent Words',
            labels={'x': 'Word', 'y': 'Frequency'},
            color=list(top_20_words.values()),
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Word cloud style table
        st.markdown("**Word Frequency Table:**")
        word_df = pd.DataFrame({
            'Word': list(top_20_words.keys()),
            'Count': list(top_20_words.values())
        })
        st.dataframe(word_df, use_container_width=True, height=500)
    
    st.markdown("---")
    
    # Word count distribution
    st.subheader("Article Length Distribution")
    
    fig = px.histogram(
        filtered_df,
        x='word_count',
        nbins=30,
        title='Distribution of Article Lengths',
        labels={'word_count': 'Word Count'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed stats table
    st.subheader("Detailed Source Statistics")
    
    source_stats = filtered_df.groupby('source').agg({
        'word_count': ['sum', 'mean', 'min', 'max'],
        'title': 'count'
    }).round(0)
    
    source_stats.columns = ['Total Words', 'Avg Words', 'Min Words', 'Max Words', 'Article Count']
    source_stats = source_stats.sort_values('Total Words', ascending=False)
    
    st.dataframe(
        source_stats.style.background_gradient(subset=['Total Words'], cmap='YlOrRd'),
        use_container_width=True
    )
