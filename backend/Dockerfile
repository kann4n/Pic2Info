# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We also download the torch packages separately to ensure compatibility
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
# Google Cloud Run automatically uses this port
EXPOSE 8000

# Define environment variable for the port
ENV PORT 8000

# Run the application using a production-grade server (gunicorn)
# -w 4: Use 4 worker processes
# -k uvicorn.workers.UvicornWorker: Use Uvicorn to run the ASGI app
# --bind 0.0.0.0:$PORT: Bind to all network interfaces on the specified port
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
