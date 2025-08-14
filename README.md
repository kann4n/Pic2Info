# Pic2Info - Image Analysis Terminal

Pic2Info is a full-stack web application that allows users to upload an image and receive a detailed analysis, including object detection, metadata extraction, and a calculated threat assessment. The interface is designed as a futuristic intelligence terminal.

## Tech Stack

**Frontend:**
- **Framework:** Next.js (React)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** Shadcn UI
- **Hosting:** Vercel

**Backend:**
- **Framework:** FastAPI
- **Language:** Python
- **ML Model:** YOLOv8n for object detection
- **Image Processing:** Pillow
- **Hosting:** Render

## Features

- **Drag & Drop Image Upload:** Easily upload images for analysis.
- **Object Detection:** Identifies objects within the image using the YOLOv8 model.
- **Metadata Extraction:** Pulls EXIF data from the image, such as dimensions, format, and device information.
- **Threat Level Assessment:** A simple algorithm to classify the threat level (Low, Medium, High) based on detected objects.
- **API Rate Limiting:** The backend API is rate-limited to 1000 requests per day per IP.

## Hosting

This application is deployed using a modern, serverless-first approach:

- The **frontend** is hosted on **Vercel**, providing a global CDN for fast delivery.
- The **backend** is hosted on **Render**, which supports Python applications and provides a persistent disk for storing the ML model file.

## How to Run Locally

To run this project on your local machine, you need to run the backend and frontend in two separate terminals.

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Create a local environment file
# (If not already created)
# Add this line to the file:
# NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

# Install Node.js dependencies
npm install

# Run the Next.js development server
npm run dev
```

Open your browser and navigate to `http://localhost:3000` to see the application.
