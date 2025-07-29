import spacy

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

NLP = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

def remove_repeated_words(cap):
    chunk_text = cap.split()
    chunk_text_no_repeats = [word for i, word in enumerate(chunk_text) if
                             (word != chunk_text[i - 1] and i > 0) or i == 0]
    chunk = ' '.join(chunk_text_no_repeats)
    return chunk

def get_nouns(nlp, captions, lemmatizer):
    all_nouns = []
    for caption in captions:
        caption_nouns = []
        tagged = nlp(caption)
        for token in tagged:
            if token.tag_ in ['NN', 'NNS']:
                caption_nouns.append(lemmatizer.lemmatize(token.text))
        all_nouns += caption_nouns
    return list(set(all_nouns))

def filter_unlikely(labels):
    hard_filter_classes, filtered = ['background', 'foreground', 'blurry', 'photo', 'photograph', 'picture', 'image',
                                     'scene', 'something', 'view', 'closeup'], []
    wn_lemmas = set(wordnet.all_lemma_names())
    colors = [
        'color', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'gray', 'brown', 'pink', 'cyan',
        'magenta', 'teal', 'lime', 'olive', 'navy', 'maroon', 'aquamarine', 'turquoise', 'silver', 'gold', 'beige',
        'coral', 'ivory', 'lavender', 'mint', 'mustard', 'peach', 'sapphire', 'scarlet', 'violet', 'indigo', 'charcoal'
    ]

    # For nonsense word filtering
    for word in labels:
        if word in wn_lemmas and word not in hard_filter_classes and len(word) > 2 and word not in colors:
            filtered.append(word)

    return filtered

def filter_captions(captions):
    nouns = get_nouns(NLP, captions, lemmatizer)
    all_nouns = filter_unlikely(nouns)
    return all_nouns