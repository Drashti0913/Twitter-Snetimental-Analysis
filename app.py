import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Fxn
def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result 

# Function to extract tweets from CSV
def extract_tweets(keyword, num_tweets):
    df = pd.read_csv("Twitter_Data.csv")
    df = df[df["clean_text"].str.contains(keyword, case=False)]
    df = df[:num_tweets]
    return df

def analyze_sentiment(text):
    sentiments = []
    for tweet in text:
        sentiment = TextBlob(tweet).sentiment.polarity
        if sentiment > 0:
            sentiments.append(('positive', sentiment))
        elif sentiment < 0:
            sentiments.append(('negative', sentiment))
        else:
            sentiments.append(('neutral', sentiment))
    return sentiments

def get_sentiment_counts(sentiments):
    pos_count = 0
    neg_count = 0
    neu_count = 0
    for sentiment in sentiments:
        if sentiment[0] == 'positive':
            pos_count += 1
        elif sentiment[0] == 'negative':
            neg_count += 1
        else:
            neu_count += 1
    return pos_count, neg_count, neu_count


def main():
    st.title("Twitter Sentiment Analyser and Visualizer")

    menu = ["Home", "Analyze from text", "Extract from Twitter", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Rules and parameters of Sentiment Analysis")

    
        css = """
        <style>
            .grey-bg {
                background-color: #f2f2f2;
                padding: 10px;
                border-radius: 5px;
            }
            .size {
                font-size: 20px;
            }
            .size1 {
                font-size: 18px;
            }

            .my-text {
                text-indent: 20px;
            }
        </style>
        """

        st.markdown(css + "<div class='grey-bg'><div class='size'>Rules for sentiment classification</div></div><div class='grey-bg'>Classification of the sentiment will be done on the following basis<br>●  +1: Positive sentiment<br>●  0: Neutral sentiment<br>●  -1: Negative sentiment</div>", unsafe_allow_html=True)
        st.markdown(css + "<div class='grey-bg'><div class='size'>Classifiers Used</div></div><div class='grey-bg'><div class='size1'>Vader Sentiment Analyzer</div><br>VaderSentiment is a lexicon-based sentiment analysis tool specifically designed to analyze social media text. It uses a rule-based approach and a lexicon of positive and negative words and phrases to calculate a sentiment score for a given text. The sentiment score ranges from -1 to 1, where -1 indicates very negative sentiment, 0 indicates neutral sentiment, and 1 indicates very positive sentiment. <br><br><div class='size1'>TextBlob</div><br>TextBlob is another popular Python library for NLP, which provides an easy-to-use interface for performing common NLP tasks like part-of-speech tagging, noun phrase extraction, sentiment analysis, etc. TextBlob's sentiment analysis is based on the Naive Bayes algorithm, which uses a training set of labeled data to learn how to classify text as positive, negative, or neutral. TextBlob's sentiment analysis also returns two values - polarity and subjectivity. Polarity is a float value between -1 and 1, where -1 indicates very negative sentiment, 0 indicates neutral sentiment, and 1 indicates very positive sentiment. Subjectivity is a float value between 0 and 1, where 0 indicates very objective text and 1 indicates very subjective text</div>", unsafe_allow_html=True)
        
    if choice == "Analyze from text":
        st.subheader("Analyze from text")
        css = """
        <style>
            .size {
                font-size: 20px;
            }
            .size1 {
                font-size: 18px;
            }

            .my-text {
                text-indent: 20px;
            }
        </style>
        """

        st.sidebar.markdown(css + "<div class='size'>Terminologies Used</div><div class='grey-bg'>Polarity<br>Polarity is a measure of the sentiment expressed in a piece of text data or a tweet using the TextBlob and vaderSentiment libraries. Range: -1 to 1.<br><br>Subjectivity<br> Subjectivity is the degree to which a data or a tweet expresses a personal opinion or feeling. Range: 0 - 1<br><br>Token sentiment<br>Token sentiment is a measure of the sentiment expressed by individual words or tokens in a data or a tweet.</div>", unsafe_allow_html=True)

        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1,col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c,use_container_width=True)

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    elif choice == "Extract from Twitter":
        st.subheader("Extract from Twitter")
        
        css = """
        <style>
            .size {
                font-size: 20px;
            }
            .size1 {
                font-size: 18px;
            }

            .my-text {
                text-indent: 20px;
            }
        </style>
        """

        st.sidebar.markdown(css + "<div class='size'>Terminologies Used</div><div class='grey-bg'>WordCloud<br>Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text.<br></div>", unsafe_allow_html=True)

    
        with st.form(key='twitterForm'):
            keyword = st.text_input("Enter keyword to search on Twitter")
            num_tweets = st.number_input("Enter number of tweets to fetch", min_value=1, max_value=1000, step=1)
            submit_button = st.form_submit_button(label='Extract') 

        if submit_button:
            # Fetch tweets
            tweets_df = extract_tweets(keyword, num_tweets)
            if not tweets_df.empty:
                # Analyze sentiment
                sentiments = analyze_sentiment(tweets_df['clean_text'])
                st.write("Tweets and their Sentiments:")

                # Get sentiment counts
                pos_count, neg_count, neu_count = get_sentiment_counts(sentiments)
                
                data = []
                for i in range(len(tweets_df)):
                    if sentiments[i][0] == 'positive':
                        data.append([tweets_df.iloc[i]['clean_text'], 'Positive'])
                    elif sentiments[i][0] == 'negative':
                        data.append([tweets_df.iloc[i]['clean_text'], 'Negative'])
                    else:
                        data.append([tweets_df.iloc[i]['clean_text'], 'Neutral'])

                table_df = pd.DataFrame(data, columns=['Tweet', 'Sentiment'])

                # set properties for the dataframe
                styles = [
                    dict(selector='th', props=[('border', '1px solid black')]),
                    dict(selector='td', props=[('border', '1px solid black')]),
                    dict(selector='th', props=[('background-color', 'lightgrey')]),
                    dict(selector='td', props=[('background-color', 'white')])
                ]

                styled_table = table_df.style\
                    .set_table_styles(styles)

                st.table(styled_table)

                # Display sentiment counts
                st.write("Sentiment count:")
                st.write(f"Positive: {pos_count}")
                st.write(f"Negative: {neg_count}")
                st.write(f"Neutral: {neu_count}")
                st.write('')

                # Display pie chart
                pie_data = {'Positive': pos_count, 'Negative': neg_count, 'Neutral': neu_count}
                pie_df = pd.DataFrame.from_dict(pie_data, orient='index', columns=['count'])
                fig = px.pie(pie_df, values='count', names=pie_df.index, title='Sentiment Distribution')
                st.plotly_chart(fig)
                
                # Concatenate all cleaned tweets
                text = ' '.join(tweets_df['clean_text'])

                # Generate the wordcloud
                st.write("Word Cloud:")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                # Display the wordcloud
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            else:
                st.warning("No tweets found.")

    if choice == "About":
        st.subheader("Meet our Developers:")

        # Define the data for the table
        table_data = pd.DataFrame({
            'Veerangi Mehta': ['20BCP003', 'Div 1, G1', 'veerangi.mce20@sot.pdpu.ac.in'],
            'Dhvanil Bhagat': ['20BCP027', 'Div 1, G1', 'dhvanil.bce20@sot.pdpu.ac.in'],
            'Drashti Bhavsar': ['20BCP040', 'Div 1, G1', 'drashti.bce20@sot.pdpu.ac.in']
        })

        # Define the CSS for the table
        table_style = """
        <style>
        table {
          border-collapse: collapse;
          width: 100%;
        }
        th, td {
          text-align: center;
          padding: 8px;
        }

        th {
          font-weight: bold;      
          background-color: #f2f2f2;
        }
        
        </style>
        """

        # Render the table without the index column
        st.markdown(table_style, unsafe_allow_html=True)
        st.table(table_data)


if __name__ == '__main__':
    main()
