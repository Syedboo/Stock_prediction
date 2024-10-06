import json
import boto3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline, logging

# Initialize S3 client
s3_client = boto3.client('s3')

# Suppress warnings from Hugging Face transformers library
logging.set_verbosity_error()

# Explicitly initialize the sentiment analysis pipeline with a specific model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def lambda_handler(event, context):
    try:
        # Part 1: Stock Data Fetching and Processing

        # Define the stock symbol and time period
        stock_symbol = 'SBRY.L'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        print(f"Start date is {start_date} and end data is {end_date}")

        # Fetch the stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        # Path to save files in the /tmp directory (ephemeral storage in Lambda)
        csv_file = '/tmp/sbrl_l_stock_data.csv'
        sorted_csv_file = '/tmp/sbrl_l_stock_data_sorted.csv'

        # Save the data to a CSV file
        stock_data.to_csv(csv_file)

        # Load stock price data from CSV
        stock_data = pd.read_csv(csv_file)

        # Ensure the date column is in datetime format
        stock_data['Date'] = pd.to_datetime(stock_data.index)

        # Drop the old index column if it exists
        stock_data.reset_index(drop=True, inplace=True)

        # Sort by date for consistency
        stock_data = stock_data.sort_values(by='Date')

        # Save the sorted data back to a CSV file
        stock_data.to_csv(sorted_csv_file, index=False)

        print(f"Stock data for {stock_symbol} has been saved to {csv_file} and sorted data saved to {sorted_csv_file}")

        # Part 2: Article Sentiment Analysis

        # Define the S3 bucket and file locations
        bucket_name = 'stockmarketprodata'
        input_key = 'Newsdata/raw/articles_data_SBRY.L.json'  # Input file path in S3
        output_key = 'Newsdata/processed/processed_articles_data_SBRY.L.json'  # Output file path in S3

        # Download the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        articles_data = response['Body'].read().decode('utf-8')
        articles = json.loads(articles_data)

        # Process articles to extract and convert dates
        for article in articles:
            if "date" in article and article["date"]:
                # Convert the date into a datetime object
                date_time = datetime.strptime(article["date"], "%a, %d %b %Y %H:%M:%S GMT")
                article["actual_date"] = date_time.date().isoformat()
                article["time"] = date_time.time().strftime("%H:%M:%S")

                # Truncate the text to the maximum length allowed by the model
                text = article["text"][:512]

                # Get sentiment analysis results
                sentiment_result = sentiment_pipeline(text)[0]

                # Convert the sentiment label to a score between 0 and 1
                sentiment_label = sentiment_result['label']
                sentiment_score = sentiment_result['score']
                if sentiment_label == "NEGATIVE":
                    sentiment_score = 1 - sentiment_score  # Invert score for negative sentiment

                # Assign sentiment score to the article
                article["sentiment_score"] = sentiment_score
                article["insights"] = sentiment_result['label']
            else:
                article["actual_date"] = None
                article["time"] = None
                article["sentiment_score"] = None
                article["insights"] = None

        # Convert the processed articles back to JSON
        processed_articles_data = json.dumps(articles, indent=4)

        # Upload the processed JSON back to S3
        s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=processed_articles_data)

        # Part 3: Merge Stock Data with Articles Data

        # Convert articles to DataFrame
        articles_df = pd.DataFrame(articles)

        # Convert the 'actual_date' column to datetime
        articles_df['actual_date'] = pd.to_datetime(articles_df['actual_date'])

        # Merge the stock data with articles data on date
        merged_df = pd.merge(stock_data, articles_df, left_on='Date', right_on='actual_date', how='left')

        # Save the merged data to a new CSV file in /tmp directory
        merged_csv_file = '/tmp/merged_stock_articles_data.csv'
        merged_df.to_csv(merged_csv_file, index=False)

        # Optionally, upload the merged data to S3
        merged_output_key = 'merged_data/merged_stock_articles_data.csv'
        s3_client.upload_file(merged_csv_file, bucket_name, merged_output_key)

        print(f"Merged stock and articles data saved to {merged_csv_file} and uploaded to {merged_output_key}")

        # Part 4: Handle Missing Values (Forward Fill and Imputation)

        # Forward fill missing values for sentiment_score and other columns
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(method='ffill')
        merged_df['insights'] = merged_df['insights'].fillna(method='ffill')
        merged_df['text'] = merged_df['text'].fillna(method='ffill')
        merged_df['title'] = merged_df['title'].fillna(method='ffill')

        # Impute remaining null values in sentiment_score with 0.5
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.5)

        # Save the updated dataset with filled missing values to /tmp
        merged_filled_csv_file = '/tmp/merged_stock_articles_data_filled_forward.csv'
        merged_df.to_csv(merged_filled_csv_file, index=False)

        # Upload the updated file to S3
        filled_output_key = 'merged_data/merged_stock_articles_data_filled_forward.csv'
        s3_client.upload_file(merged_filled_csv_file, bucket_name, filled_output_key)

        print(f"Missing values filled and updated data saved to {merged_filled_csv_file} and uploaded to {filled_output_key}")

        return {
            'statusCode': 200,
            'body': json.dumps('Stock data processed, articles analyzed, merged, missing values handled, and saved successfully!')
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing: {str(e)}')
        }
