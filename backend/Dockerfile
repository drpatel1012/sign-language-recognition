FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app
COPY backend/app.py /app/backend/app.py

# Copy the trained model
COPY model.pkl /app/model.pkl

# Expose port and run the app
EXPOSE 5000
CMD ["python", "backend/app.py"]
