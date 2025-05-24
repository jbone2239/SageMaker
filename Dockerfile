FROM python:3.8-slim

# Set up directories
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /opt/program

# Install dependencies
COPY model.joblib .
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy code
COPY wine_lr_model.py .
COPY inference.py .

# Specify how the container serves requests
ENV SAGEMAKER_PROGRAM=inference.py

CMD ["python", "inference.py"]
