FROM python:3.10

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY ["app.py", "requirements.txt", "/app/"]
COPY ["model/", "/app/model/"]
# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 80
EXPOSE 8080

# Run app.py at container launch
CMD ["python", "app.py"]