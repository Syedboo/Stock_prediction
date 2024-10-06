# Use an official AWS Lambda base image for Python 3.8
FROM public.ecr.aws/lambda/python:3.8

# Copy the function code into the container
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Copy requirements.txt and install dependencies
COPY requirements.txt ./

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install PyTorch from the official repository (latest stable version for CPU) and Hugging Face's Transformers
RUN pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install transformers

# Set the Hugging Face cache directory to /tmp to make it writable by Lambda
ENV TRANSFORMERS_CACHE=/tmp

# Pre-download the Hugging Face model at build time to avoid downloading it every time the Lambda function is invoked
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Command to run your Lambda function
CMD ["lambda_function.lambda_handler"]
