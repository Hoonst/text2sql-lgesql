conda create -n text2sql python=3.6
source activate text2sql
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -c "import stanza; stanza.download('en')"
python -c "from embeddings import GloveEmbedding; emb = GloveEmbedding('common_crawl_48', d_emb=300)"
python -c "import nltk; nltk.download('stopwords')"