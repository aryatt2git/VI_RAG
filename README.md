# VI_RAG
Retrieval Augmented Generation system to support Variant Interpretation

1. Activate conda environment.
```bash
conda activate VI_RAG
```
2. Install Ollama.
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
3. Pull the model you want to use via Ollama.
```bash
ollama pull gpt-oss:120b-cloud
```
4. Install additional dependencies using pip.
```bash
pip install -r requirements.txt
```
5. Compose the docker image in the weaviate directory.
```bash
cd weaviate
docker compose up -d
```

don't forget to:
brew install poppler
brew install tesseract
brew install tesseract-lang