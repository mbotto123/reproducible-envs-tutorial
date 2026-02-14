FROM python:3.12

RUN pip install --no-cache-dir numpy==2.4.2 scipy==1.17.0

# Copy source code and input file into the image
COPY ilu_study.py laplacian_2d_matrix.npz /home

WORKDIR /home

# Get help text from script by default
CMD ["python", "ilu_study.py", "-h"]
