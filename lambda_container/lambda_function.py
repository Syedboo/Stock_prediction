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

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def upload_to_s3(local_file, bucket_name, s3_key):
    try:
        # Delete the file if it exists in S3 (ensure no duplicates)
        delete_from_s3(bucket_name, s3_key)
        
        # Upload the new file
        s3_client.upload_file(local_file, bucket_name, s3_key)
        print(f"Uploaded {local_file} to {s3_key}")
    except Exception as e:
        print(f"Error uploading {local_file} to {s3_key}: {str(e)}")

def delete_from_s3(bucket_name, s3_key):
    try:
        # Delete the object if it exists in S3
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Deleted {s3_key} from {bucket_name} if it existed.")
    except Exception as e:
        print(f"Error deleting {s3_key} from {bucket_name}: {str(e)}")

def lambda_handler(event, context):
    try:
        # Part 1: Stock Data Fetching and Processing
        stock_symbol = 'SBRY.L'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Fetch the stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)  # Reset index to ensure 'Date' is a column
        csv_file = '/tmp/sbrl_l_stock_data.csv'
        stock_data.to_csv(csv_file)

        # Upload stock data to S3, ensure no duplicates
        bucket_name = 'stockmarketprodata'
        stock_output_key = 'stock_data/raw_sbrl_l_stock_data.csv'
        upload_to_s3(csv_file, bucket_name, stock_output_key)

        # Part 2: Article Sentiment Analysis
        input_key = 'Newsdata/raw/articles_data_SBRY.L.json'
        output_key = 'Newsdata/processed/processed_articles_data_SBRY.L.json'

        # Download the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        articles = json.loads(response['Body'].read().decode('utf-8'))

        # Process articles and sentiment analysis
        for article in articles:
            if "date" in article and article["date"]:
                date_time = datetime.strptime(article["date"], "%a, %d %b %Y %H:%M:%S GMT")
                article["actual_date"] = date_time.date().isoformat()

                # Truncate text and get sentiment
                text = article["text"][:512]
                sentiment_result = sentiment_pipeline(text)[0]

                # Adjust sentiment score
                sentiment_score = sentiment_result['score']
                if sentiment_result['label'] == "NEGATIVE":
                    sentiment_score = 1 - sentiment_score

                article["sentiment_score"] = sentiment_score
                article["insights"] = sentiment_result['label']

        # Save processed articles to S3
        processed_articles_file = '/tmp/processed_articles_data.json'
        with open(processed_articles_file, 'w') as f:
            json.dump(articles, f, indent=4)
        upload_to_s3(processed_articles_file, bucket_name, output_key)

        # Part 3: Merge Stock Data with Articles Data
        articles_df = pd.DataFrame(articles)
        articles_df['actual_date'] = pd.to_datetime(articles_df['actual_date'])

        # Convert 'Date' in stock_data to datetime format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # Merge stock data with articles data
        merged_df = pd.merge(stock_data, articles_df, left_on='Date', right_on='actual_date', how='left')

        # Save merged data to S3
        merged_csv_file = '/tmp/merged_stock_articles_data.csv'
        merged_df.to_csv(merged_csv_file, index=False)
        merged_output_key = 'merged_data/merged_stock_articles_data.csv'
        upload_to_s3(merged_csv_file, bucket_name, merged_output_key)

        # Part 4: Handle Missing Values (Forward Fill and Imputation)
        # Forward fill missing values for sentiment_score and other columns
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(method='ffill')
        merged_df['insights'] = merged_df['insights'].fillna(method='ffill')
        merged_df['text'] = merged_df['text'].fillna(method='ffill')
        merged_df['title'] = merged_df['title'].fillna(method='ffill')

        # Impute remaining null values in sentiment_score with 0.5
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.5)

        # Convert 'Date' column to datetime and drop rows where conversion fails
        merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')

        # Drop rows where 'Date' is NaT (invalid dates)
        merged_filled_forward = merged_df.dropna(subset=['Date'])

        # Save the updated dataset with filled missing values to /tmp and upload to S3
        merged_filled_csv_file = '/tmp/merged_stock_articles_data_filled_forward.csv'
        merged_filled_forward.to_csv(merged_filled_csv_file, index=False)
        filled_output_key = 'merged_data/merged_stock_articles_data_filled_forward.csv'
        upload_to_s3(merged_filled_csv_file, bucket_name, filled_output_key)

        print(f"Invalid dates removed, missing values filled, and updated data saved to {merged_filled_csv_file} and uploaded to {filled_output_key}")

        return {
            'statusCode': 200,
            'body': json.dumps('Stock data processed, articles analyzed, merged, missing values handled, and saved successfully!')
        }

    except Exception as e:
        # Catch any errors and return them as a response
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing data: {str(e)}")
        }
