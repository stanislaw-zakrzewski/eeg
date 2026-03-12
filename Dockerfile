# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p /root/mne_data
RUN mkdir -p bids_dataset

# Install Poetry
RUN pip install poetry

# Copy the project files into the container
COPY . .

# Install project dependencies
# --no-root is used because we are installing dependencies for the project itself, not to be used as a library
RUN poetry install --no-root

# Set the entrypoint to run the main script
ENTRYPOINT ["poetry", "run", "python", "server.py"]
