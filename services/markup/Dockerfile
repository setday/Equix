FROM pytorch/pytorch

ARG DEBIAN_FRONTEND=noninteractive

### Set the working directory
WORKDIR /app

### Copy Service files
COPY services/markup/serve_web_tunnel.py /app/app.py

### Install GCC for poppler
RUN apt-get update && \
    apt-get install -y g++

### Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        python-dotenv==1.1.0 \
        fastapi==0.115.12 \
        pdf2image==1.17.0 \
        pydantic==2.10.5 \
        pydantic-settings==2.7.1 \
        transformers==4.48.0 \
        uvicorn==0.34.2 \
        python-multipart==0.0.20 \
        timm==1.0.15 \
        python-poppler==0.4.1

### Expose the port
EXPOSE 5123
