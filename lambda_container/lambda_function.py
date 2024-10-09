import json
import boto3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline, logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# Initialize S3 client
s3_client = boto3.client('s3')

# Suppress warnings from Hugging Face transformers library
logging.set_verbosity_error()

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to upload a file to S3
def upload_to_s3(local_file, bucket_name, s3_key):
    try:
        # Delete the file if it exists in S3 (ensure no duplicates)
        delete_from_s3(bucket_name, s3_key)
        
        # Upload the new file
        s3_client.upload_file(local_file, bucket_name, s3_key)
        print(f"Uploaded {local_file} to {s3_key}")
    except Exception as e:
        print(f"Error uploading {local_file} to {s3_key}: {str(e)}")

# Function to delete a file from S3
def delete_from_s3(bucket_name, s3_key):
    try:
        # Delete the object if it exists in S3
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Deleted {s3_key} from {bucket_name} if it existed.")
    except Exception as e:
        print(f"Error deleting {s3_key} from {bucket_name}: {str(e)}")


# Function to forecast the next `days` using Random Forest model
def forecast_next_days_rf(new_sentiment_score, model, initial_data, features, imputer, days=1): 
    forecasts = []
    data = initial_data.copy()

    for _ in range(days):
        # Prepare features for prediction
        latest_features = data[features].values

        # Ensure no NaN values
        latest_features = imputer.transform(latest_features)

        # Predict the next day's price
        next_day_price = model.predict(latest_features)[0]
        forecasts.append(next_day_price)

        # Update the data with the new prediction
        new_row = data.iloc[-1:].copy()
        new_row['Open'] = next_day_price
        new_row['sentiment_lag_1'] = new_sentiment_score
        new_row['sentiment_lag_2'] = data['sentiment_score'].iloc[-1]
        new_row['sentiment_roll_mean_3'] = data['sentiment_score'].iloc[-3:].mean()
        new_row['sentiment_roll_std_3'] = data['sentiment_score'].iloc[-3:].std()
        new_row['open_roll_mean_3'] = data['Open'].iloc[-3:].mean()
        new_row['open_roll_std_3'] = data['Open'].iloc[-3:].std()
        new_row['MA10'] = data['Open'].iloc[-10:].mean()
        new_row['MA50'] = data['Open'].iloc[-50:].mean()
        new_row['Volatility'] = data['Open'].iloc[-10:].std()

        # Append new_row to data
        data = pd.concat([data, new_row], ignore_index=True)

    return forecasts

sns_client = boto3.client('sns')

# Function to send forecast results to SNS
def send_forecast_to_sns(rf_mse, topic_arn):
    try:
        # Create the message with additional text
        message = f"The forecast is complete. The calculated Root Mean Square Error (RMSE) for the random forest model is: {rf_mse:.4f}. This is the prediction accuracy measure."

        # Send message to SNS
        response = sns_client.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject='Stock Price Prediction Result'
        )

        print(f"Forecast sent to SNS: {message}")
        return response
    except Exception as e:
        print(f"Error sending forecast to SNS: {str(e)}")
        return None

def lambda_handler(event, context):
    try:
        #subscribing to the topic i.e) adding user to the subscription list to send email
        body = json.loads(event['body'])
        stock_symbol = body.get('stockSymbol')
        email = body.get('email')
        response = sns_client.subscribe(
            TopicArn='arn:aws:sns:ap-south-1:975050245649:Lambda_to_email',
            Protocol='email',
            Endpoint=email
        )
        subscription_arn = response['SubscriptionArn']
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
        articles_df = articles_df[['actual_date', 'text', 'sentiment_score', 'insights']]  # Select only relevant columns
        articles_df['actual_date'] = pd.to_datetime(articles_df['actual_date'])

        # Convert 'Date' in stock_data to datetime format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # Merge stock data with articles data
        merged_df = pd.merge(stock_data, articles_df, left_on='Date', right_on='actual_date', how='left')

        # Select only the required columns
        merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'sentiment_score']]

        # Part 4: Handle Missing Values Before Feature Engineering
        numeric_cols = merged_df.select_dtypes(include=[np.float64, np.int64]).columns  # Only numerical columns

        # Impute missing values for numeric columns before adding new features
        imputer = SimpleImputer(strategy='mean')
        merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])

        # Part 5: Feature Engineering: Adding Technical Indicators
        merged_df['sentiment_lag_1'] = merged_df['sentiment_score'].shift(1)
        merged_df['sentiment_lag_2'] = merged_df['sentiment_score'].shift(2)
        merged_df['sentiment_roll_mean_3'] = merged_df['sentiment_score'].rolling(window=3).mean()
        merged_df['sentiment_roll_std_3'] = merged_df['sentiment_score'].rolling(window=3).std()
        merged_df['open_roll_mean_3'] = merged_df['Open'].rolling(window=3).mean()
        merged_df['open_roll_std_3'] = merged_df['Open'].rolling(window=3).std()

        # Adding Moving Averages
        merged_df['MA10'] = merged_df['Open'].rolling(window=10).mean()
        merged_df['MA50'] = merged_df['Open'].rolling(window=50).mean()

        # Adding Volatility
        merged_df['Volatility'] = merged_df['Open'].rolling(window=10).std()

        # Ensure no new missing values after feature engineering (impute again if needed)
        new_numeric_cols = merged_df.select_dtypes(include=[np.float64, np.int64]).columns
        merged_df[new_numeric_cols] = imputer.fit_transform(merged_df[new_numeric_cols])

        # Save the dataset to S3
        final_csv_file = '/tmp/final_stock_price_dataset.csv'
        merged_df.to_csv(final_csv_file, index=False)
        final_output_key = 'merged_data/final_stock_price_dataset.csv'
        upload_to_s3(final_csv_file, bucket_name, final_output_key)

        # Part 6: Train-Test Split and Model Training
        features = ['sentiment_score', 'sentiment_lag_1', 'sentiment_lag_2', 'sentiment_roll_mean_3', 'sentiment_roll_std_3',
                    'open_roll_mean_3', 'open_roll_std_3', 'MA10', 'MA50', 'Volatility']
        target = 'Open'

        # Train-test split (80% train, 20% test)
        train_size = int(len(merged_df) * 0.8)
        train = merged_df[:train_size]
        test = merged_df[train_size:]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        # Model Training with Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # Model Evaluation
        rf_mse = mean_squared_error(y_test, y_pred_rf)
        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        rf_rmse = np.sqrt(rf_mse)

        # Send forecast result to SNS
        topic_arn = 'arn:aws:sns:ap-south-1:975050245649:Lambda_to_email'
        send_forecast_to_sns(rf_mse, topic_arn)
        print(f"Random Forest MSE: {rf_mse}, MAE: {rf_mae}, RMSE: {rf_rmse}")

        return {
            'statusCode': 200,
            'body': json.dumps('Stock data processed, articles analyzed, features engineered, model trained, and saved successfully!')
        }

    except Exception as e:
        # Catch any errors and return them as a response
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing data: {str(e)}")
        }

