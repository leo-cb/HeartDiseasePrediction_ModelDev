# Use the official Python image as a base image
FROM python:3.11.4

# Set the working directory within the container
WORKDIR /app

# Reload local package database
RUN apt-get update

# Copy files to container
COPY . /app

# Install the Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt