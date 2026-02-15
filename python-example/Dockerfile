FROM python:3.12

WORKDIR /usr/src/app

# Install Python dependencies
COPY requirements.txt /usr/src/app
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and input file into the image
COPY ilu_study.py laplacian_2d_matrix.npz /usr/src/app

# Get help text from script by default
CMD ["python", "ilu_study.py", "-h"]
