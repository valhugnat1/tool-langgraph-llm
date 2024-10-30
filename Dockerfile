# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app


# Install system dependencies first
RUN apt-get update \
    && apt-get -y install libpq-dev gcc 

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages in the virtual environment
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PORT 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]