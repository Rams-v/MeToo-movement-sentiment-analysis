import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import spacy
import re
import html
import contractions
analyzer = SentimentIntensityAnalyzer()
from collections import Counter

# Disable the showPyplotGlobalUse warning
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #FFBFAF;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Declare df outside the functions
df = None

# Function to upload the dataset
def upload_dataset():
    global df  

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Please upload a CSV file.")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            # Delete the 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
            required_columns = ['text', 'favoriteCount', 'retweetCount']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if not missing_columns:
                st.success("Dataset uploaded successfully")
                st.subheader("Preview of the Dataset:")
                st.dataframe(df.head(10).replace({True: 'True', False: 'False'}), use_container_width=True)
            else:
                st.error("The dataset is missing the following required columns: " + ", ".join(missing_columns))
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {str(e)}")

        st.header("Basic Analysis of the Dataset")

        # Display dataset information
        st.write("Dataset Information:")
        st.write(df.describe())
        
        # Unique tweets analysis
        st.write("Unique Tweet Analysis")
        unique_tweets = {"Total Tweets": len(df), "Unique Tweets": len(df['text'].unique())}
        
        # Plot the unique tweet count
        plt.figure(figsize=(9, 9))
        bar = plt.bar(x=unique_tweets.keys(), height=unique_tweets.values(), color=['#FFC0C3', '#FFA0A4'])
        for bar, count in zip(bar, unique_tweets.values()):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 5000, str(count), fontsize=15)
        custom_yticks = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]
        custom_ylabels = ['50K', '100K', '150K', '200K', '250K', '300K', '350K', '400K']
        plt.yticks(custom_yticks, custom_ylabels, fontsize=15)
        plt.xticks(fontsize=15)
        plt.title('Unique Tweet Count', fontsize=25)
        st.pyplot()
        
        # Truncated Tweets
        st.write("Truncated Tweet Distribution")
        truncated_tweets = pd.DataFrame(df['truncated'].value_counts())
        
        # Plot the Truncated Tweet Distribution
        plt.figure(figsize=(9, 9))
        plt.bar(x=truncated_tweets.index, height=truncated_tweets['truncated'], color=['#FFC0C3', '#FFA0A4'])
        custom_yticks_3 = [25000, 50000, 75000, 100000, 125000]
        custom_ylabels_4 = ['25K', '50K', '75K', '100K', '125K']
        plt.yticks(custom_yticks_3, custom_ylabels_4, fontsize=15)
        plt.xticks([0, 1], ['False', 'True'], fontsize=18)
        plt.title('Truncated Tweet Distribution', fontsize=20)
        st.pyplot()
        
        # Reply vs. Original Tweets
        st.write("Reply vs Original Tweets")
        not_null_count = df['replyToSN'].count()
        null_count = df['replyToSN'].isnull().sum()
        reply_tweets = pd.DataFrame({'Status': ['Reply Tweet', 'Original Tweet'], 'Value': [not_null_count, null_count]})
        
        # Plot Reply vs. Original Tweets
        plt.figure(figsize=(9, 9))
        plt.bar(x=reply_tweets['Status'], height=reply_tweets['Value'], color=['#FFC0C3', '#FFA0A4'])
        custom_yticks_5 = [25000, 50000, 75000, 100000, 125000, 150000]
        custom_ylabels_5 = ['25K', '50K', '75K', '100K', '125K', '150K']
        plt.yticks(custom_yticks_5, custom_ylabels_5, fontsize=15)
        plt.xticks(fontsize=18)
        plt.title('Reply vs. Original Tweets', fontsize=20)
        st.pyplot()
        
        # Tweets over time
        df['created'] = pd.to_datetime(df['created'])
        df['date'] = df['created'].dt.date
        tweet_counts = df.groupby(df['date'])['text'].count()
        fig = px.line(tweet_counts, x=tweet_counts.index, y='text', markers=True, line_shape='linear',
                     title='Number of Tweets Over Time')
        fig.update_layout(
            plot_bgcolor='#FCE3E4',
            paper_bgcolor='white',
        )
        fig.update_traces(line=dict(color='#541F13'))
        fig.update_xaxes(title='Date', tickangle=-45, showgrid=True)
        fig.update_yaxes(title='Number of Tweets', showgrid=True)
        st.write("Tweets Over Time:")
        st.plotly_chart(fig)

        # Unique vs. Original Tweets
        df1 = df.drop_duplicates(subset='text', keep='first')
        retweet_counts = pd.DataFrame(df1['isRetweet'].value_counts())
        st.write("Unique Retweet vs Original Tweet Count")
        
        # Plot the Unique Retweet vs. Original Tweet Count
        plt.figure(figsize=(9, 9))
        plt.bar(x=retweet_counts.index, height=retweet_counts['isRetweet'], color=['#FFC0C3', '#FFA0A4'])
        custom_yticks_1 = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
        custom_ylabels_2 = ['25K', '50K', '75K', '100K', '125K', '150K', '175K', '200K']
        plt.yticks(custom_yticks_1, custom_ylabels_2, fontsize=15)
        plt.xticks([0, 1], ['False', 'True'], fontsize=18)
        plt.title('Unique Retweet vs. Original Tweet Count', fontsize=20)
        st.pyplot()

        unique_people_count = len(df['replyToSN'].unique())
        st.write("Number of Unique People in the 'replyToSN' Column:", unique_people_count)

        replyto = pd.DataFrame(df1['replyToSN'].value_counts())
        username_counts = dict(zip(replyto.index, replyto['replyToSN']))

        # Generate a word cloud of Twitter handles
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(username_counts)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title("Word Cloud of Twitter Handles in the 'replyTo' Column", fontsize=20)
        plt.axis('off')
        st.pyplot()
    else:
        st.warning("Please upload a dataset first.")
        
def preprocess_text(text):
    # Decoding correction for special characters
    text = html.unescape(text)

    # Removal of URLs
    text = re.sub(r'https?://\S+', '', text)

    # Removal of usernames (Twitter handles)
    text = re.sub(r'@\w+', '', text)

    # Removal of hashtags
    text_without_hashtags = re.sub(r'#\w+', '', text)

    # Removal of "RT: " (commonly found in retweets)
    text_without_rt = text_without_hashtags.replace('RT : ', '')

    # Removal of leading and trailing spaces
    text_stripped = text_without_rt.strip()

    # Removal of trailing underscores
    text_no_underscore = text_stripped.rstrip('_')

    # Removal of leading and trailing double quotes
    text_no_quotes = text_no_underscore.strip('"')

    # Conversion to lowercase
    text_lower = text_no_quotes.lower()

    # Expansion of contractions
    expanded_text = contractions.fix(text_lower)

    # Removal of punctuation
    exclude = string.punctuation
    text_no_punctuation = expanded_text.translate(str.maketrans('', '', exclude))

    return text_no_punctuation

def sentiment_analysis():
    st.title("Sentiment Analysis")
    tweet_input = st.text_area("Enter a tweet for sentiment analysis:")

    if st.button("Analyze Sentiment"):
        preprocessed_tweet = preprocess_text(tweet_input)
        sentiment = analyzer.polarity_scores(preprocessed_tweet)
        if sentiment['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        st.write("Preprocessed Tweet:", preprocessed_tweet)
        st.write("Sentiment Score:", sentiment)
        st.write("Sentiment Label:", sentiment_label)
def generate_word_cloud(text, title):
    text = text.astype(str)  # Ensure the text is treated as a string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)  # Show the Matplotlib plot in Streamlit

def eda():
    st.title("Exploratory Data Analysis (EDA)")
    option = st.selectbox("Select an Option", ["Preprocessing", "Sentiment Analysis"])

    if option == "Preprocessing":
        uploaded_file = st.file_uploader("Upload a CSV file for Preprocessing EDA", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.write("Exploring Tweets with highest Favorite Count")
            sorted_df = df.sort_values(by='favoriteCount', ascending=False)
            top_tweets = sorted_df.head()
            st.write(top_tweets)
            

            st.write("Timeline of the 1000 Most Favorited Tweets")
            if 'created' in df.columns:
                df_sorted_rt = df.sort_values(by='created', ascending=True)
                top_100_rt = df_sorted_rt.head(1000)
                top_100_rt['created'] = pd.to_datetime(top_100_rt['created'])

                pink_color = "#FF6B6B"

                fig = px.scatter(
                    top_100_rt,
                    x='created',
                    y='favoriteCount',
                    title='Timeline of the 1000 Most Favorited Tweets',
                    labels={'created': 'Date and Time', 'favoriteCount': 'Favorite Count'},
                    color_discrete_sequence=[pink_color]
                )

                fig.update_xaxes(title_text='Date and Time')
                fig.update_yaxes(title_text='Favorite Count')

                fig.update_xaxes(
                    tickmode='linear',
                    dtick='D1',
                    tickformat='%Y-%m-%d',
                )
                st.plotly_chart(fig)
            else:
                st.write("The 'created' column is missing in your dataset.")

            
            

            if 'replyToSN' in df.columns:
                df_replies = df.dropna(subset=['replyToSN'])
                mention_counts = Counter(df_replies['replyToSN'])
                most_popular_people = mention_counts.most_common(10)  # Get the top 10 most mentioned people

                # Create a DataFrame with the results
                most_replied_df = pd.DataFrame(most_popular_people, columns=['Person', 'Replies'])

                st.write("Most Replied-To People:")
                st.write(most_replied_df)
            else:
                st.write("The 'replyToSN' column is missing in your dataset.")

        else:
            st.write("Please upload a CSV file for Preprocessing EDA.")

    elif option == "Sentiment Analysis":
        uploaded_file = st.file_uploader("Upload a CSV file for Sentiment Analysis EDA", type=["csv"])
        if uploaded_file is not None:
            df1 = pd.read_csv(uploaded_file)
            sentiment_counts = df1['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment Label', 'Count']

            colors = {
                'Positive': '#FFB6C1',
                'Negative': '#8B4513',
                'Neutral': '#F5F5DC',
            }

            fig = px.bar(sentiment_counts, x='Sentiment Label', y='Count', title='Sentiment Distribution', color='Sentiment Label', color_discrete_map=colors, text='Count')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig)

            st.write("Word Cloud of the Sentiments")
            positive_tweets = df1[df1['sentiment_label'] == 'Positive']
            negative_tweets = df1[df1['sentiment_label'] == 'Negative']
            neutral_tweets = df1[df1['sentiment_label'] == 'Neutral']
            st.write("Positive Word Cloud")
            generate_word_cloud(positive_tweets['punctuation_removed'], 'Positive Sentiment Word Cloud')
            
            st.write("Negative Word Cloud")
            generate_word_cloud(negative_tweets['punctuation_removed'], 'Negative Sentiment Word Cloud')
            
            st.write("Neutral Word Cloud")
            generate_word_cloud(neutral_tweets['punctuation_removed'], 'Neutral Sentiment Word Cloud')
        else:
            st.write("Please upload a CSV file for Sentiment Analysis EDA.")
def about():
    st.markdown("""
    <div style="font-size: 18px; color: #541F13;">
        The #MeToo movement is a global social and cultural phenomenon that emerged in the early 21st century. 
        It began as a grassroots campaign on social media platforms and quickly gained momentum, 
        with the primary goal of raising awareness about and combatting sexual harassment and sexual assault, 
        particularly in the workplace. The movement encourages individuals, primarily women, to share their 
        personal experiences of harassment and abuse, often using the hashtag #MeToo. These testimonies shed 
        light on the pervasive nature of such incidents and challenge the prevailing culture of silence and 
        victim-blaming. The #MeToo movement has had a profound impact on public discourse, leading to 
        increased awareness, legal changes, and the removal of powerful figures in various industries accused 
        of misconduct. It has sparked important conversations about consent, gender inequality, and the need 
        for systemic change in addressing and preventing sexual harassment. While the movement has made 
        significant strides in addressing these issues, it also underscores the ongoing work required to create 
        a safer and more equitable world for all.
    </div>
    <br>
    """, unsafe_allow_html=True)



# Define the content for each page in a dictionary
page_content = {
    'About': about,
    'Basic Analysis': upload_dataset,
    'Sentiment Analysis':sentiment_analysis,
    'Exploratory Data Analysis':eda
}

# Add the image at the top of the page (outside of the content dictionary)
st.image("Header.png", use_column_width=True)

# Set the sidebar theme
st.sidebar.markdown(
    f"""
    <style>
    .sidebar-container {{
        background-color: #FFFBF1;
        padding: 10px;
        border-radius: 5px;
    }}
    .sidebar-button {{
        background-color: transparent;
        color: #541F13;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
        width: 100%;
        text-align: left;
    }}
    .sidebar-button:hover {{
        background-color: #A46BF5;
        transform: scale(1.05);
    }}
    .sidebar-rectangle {{
        background-color: #FFFBF1;
        height: 20px;
        width: 100%;
        border: none;
        border-radius: 0;
        margin: 0;
        padding: 0;
        cursor: default;
    }}
    .sidebar-bottom-buttons {{
        display: flex;
        justify-content: space-between;
        padding: 5px 10px;
    }}
    .sidebar-bottom-button {{
        background-color: #CD816F;  /* Change the background color */
        color: #FFFBF1;  /* Change the text color */
        padding: 5px 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }}
    .sidebar-bottom-button:hover {{
        background-color: #FFFBF1;
        color: #CD816F;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Display logo in the sidebar
st.sidebar.image("logo.png", use_column_width=True)

# Create clickable buttons for each page in the sidebar
selected_page = st.sidebar.radio("Select a Page", list(page_content.keys()))

# Add a rectangle at the bottom of the sidebar
st.sidebar.markdown("<div class='sidebar-rectangle'></div>", unsafe_allow_html=True)

# Add logos to login and helpline buttons at the bottom of the sidebar
st.sidebar.markdown(
    """
    <div class='sidebar-bottom-buttons'>
        <button class='sidebar-bottom-button' onclick=''>Login</button>
        <button class='sidebar-bottom-button' onclick=''>Helpline</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Main function to display page content based on the selected page
def main():
    if selected_page == 'About':
        about()
    elif selected_page == 'Basic Analysis':
        upload_dataset()
    elif selected_page=='Sentiment Analysis':
        sentiment_analysis()
    elif selected_page=="Exploratory Data Analysis":
        eda()

# Call the main function to display the selected page
main()
