FROM python:3.11-slim
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace
RUN pip install --no-cache-dir -e .[dev] || pip install --no-cache-dir -e .
CMD ["python", "-m", "eigen_jepa.train", "--help"]
