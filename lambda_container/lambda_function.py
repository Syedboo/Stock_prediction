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

def upload_to_s3(local_file, bucket_name, s3_key):
    try:
        # Delete the file if it exists in S3
        delete_from_s3(bucket_name, s3_key)
        
        # Upload the new file
        s3_client.upload_file(local_file, bucket_name, s3_key)
        print(f"Uploaded {local_file} to {s3_key}")
    except Exception as e:
        print(f"Error uploading {local_file} to {s3_key}: {str(e)}")

def delete_from_s3(bucket_name, s3_key):
    try:
        # Check if the object exists and delete it
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Deleted {s3_key} from {bucket_name} if it existed.")
    except Exception as e:
        print(f"Error deleting {s3_key} from {bucket_name}: {str(e)}")

def lambda_handler(event, context):
    try:
        # Part 1: Stock Data Fetching and Processing

        # Define the stock symbol and time period
        stock_symbol = 'SBRY.L'
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        print(f"Start date is {start_date} and end date is {end_date}")

        # Fetch the stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        # Path to save files in the /tmp directory (ephemeral storage in Lambda)
        csv_file = '/tmp/sbrl_l_stock_data.csv'

        # Save the stock data to a CSV file
        stock_data.to_csv(csv_file)

        # Upload the stock data CSV to S3 after fetching it from yfinance
        bucket_name = 'stockmarketprodata'
        stock_output_key = 'stock_data/raw_sbrl_l_stock_data.csv'
        upload_to_s3(csv_file, bucket_name, stock_output_key)

        # Load stock price data from CSV
        stock_data = pd.read_csv(csv_file)

        # Ensure the date column is in datetime format
        stock_data['Date'] = pd.to_datetime(stock_data.index)

        # Drop the old index column if it exists
        stock_data.reset_index(drop=True, inplace=True)

        # Save the updated data back to a CSV file after resetting the index and ensuring date format
        updated_csv_file = '/tmp/updated_sbrl_l_stock_data.csv'
        stock_data.to_csv(updated_csv_file, index=False)

        # Upload the updated stock data CSV to S3
        updated_stock_output_key = 'stock_data/updated_sbrl_l_stock_data.csv'
        upload_to_s3(updated_csv_file, bucket_name, updated_stock_output_key)

        # Part 2: Article Sentiment Analysis

        # Define the S3 bucket and file locations
        input_key = 'Newsdata/raw/articles_data_SBRY.L.json'
        output_key = 'Newsdata/processed/processed_articles_data_SBRY.L.json'

        # Download the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=input_key)
        articles_data = response['Body'].read().decode('utf-8')
        articles = json.loads(articles_data)

        # Upload the raw articles data to S3 before processing
        raw_articles_file = '/tmp/raw_articles_data.json'
        with open(raw_articles_file, 'w') as f:
            json.dump(articles, f)
        upload_to_s3(raw_articles_file, bucket_name, 'Newsdata/raw/raw_articles_data.json')

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

        # Convert the processed articles back to JSON and upload to S3 after sentiment analysis
        processed_articles_file = '/tmp/processed_articles_data.json'
        with open(processed_articles_file, 'w') as f:
            json.dump(articles, f, indent=4)
        upload_to_s3(processed_articles_file, bucket_name, output_key)

        # Part 3: Merge Stock Data with Articles Data

        # Convert articles to DataFrame
        articles_df = pd.DataFrame(articles)

        # Convert the 'actual_date' column to datetime
        articles_df['actual_date'] = pd.to_datetime(articles_df['actual_date'])

        # Merge the stock data with articles data on date
        merged_df = pd.merge(stock_data, articles_df, left_on='Date', right_on='actual_date', how='left')

        # Save the merged data to a new CSV file and upload to S3
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

        # Save the updated dataset with filled missing values to /tmp and upload to S3
        merged_filled_csv_file = '/tmp/merged_stock_articles_data_filled_forward.csv'
        merged_df.to_csv(merged_filled_csv_file, index=False)
        filled_output_key = 'merged_data/merged_stock_articles_data_filled_forward.csv'
        upload_to_s3(merged_filled_csv_file, bucket_name, filled_output_key)

        print(f"Missing values filled and updated data saved to {merged_filled_csv_file} and uploaded to {filled_output_key}")

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
