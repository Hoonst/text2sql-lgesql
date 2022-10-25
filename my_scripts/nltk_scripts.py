# import nltk
# nltk.download('punkt')

# nltk 'punkt'가 다운로드가 되지 않을때 사용하는 script


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')