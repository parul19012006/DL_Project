"""
app.py  —  Hinglish NLP Sentiment Analyzer
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords, words as nltk_words
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import pandas as pd

app = Flask(__name__, static_folder=".")

# ── Dataset ──────────────────────────────────────────────────────────────────
data = {
    'text': [
        "Yaar ye movie bahut hi amazing thi, must watch hai!",
        "Aaj ka din bahut productive raha, sab kuch on time hua",
        "Bhai ye restaurant ka khana literally best hai poore city mein",
        "Finally exam clear ho gaya, bohot khush hoon aaj",
        "Job offer aa gaya bhai! Dream company se call aaya",
        "Dost log bahut caring hain, I'm so lucky to have them",
        "Aaj gym mein beast mode tha, feeling so energetic",
        "Ye naya phone ekdum zabardast hai, camera is outstanding",
        "Match mein India ne jeeta! Kya performance tha yaar",
        "Promotion mil gayi finally, saal bhar ki mehnat rang layi",
        "Weekend trip planning kar rahe hain, super excited hoon",
        "Ghar ka khana hamesha best rehta hai, nothing beats it",
        "Didi ne surprise gift diya, so thoughtful hai woh",
        "Coffee peete hue sunset dekha, best feeling ever",
        "Ye song mujhe bohot pasand hai, repeat pe sun raha hoon",
        "Yaar aaj traffic mein 2 ghante phase raha, bahut bura laga",
        "Exam mein bahut kharab gaya, disappointed hoon khud se",
        "Bhai boss ne phir se overtime bola, this is so unfair",
        "Phone kho gaya, itna important data tha usme, devastated hoon",
        "Dosto ne plan cancel kar diya last minute pe, bahut bura laga",
        "Ye internet itna slow hai, literally pagal ho jaunga",
        "Kuch bhi sahi nahi ho raha aajkal, everything feels wrong",
        "Salary cut ho gayi, bohot tension ho rahi hai yaar",
        "Ye movie waste of time thi, poora paisa barbaad",
        "Aaj baarish mein bheeg gaya aur koi umbrella nahi tha",
        "Relationship mein bahut problems aa rahi hain recently",
        "Ghar mein lights gayi aur important meeting thi, disaster",
        "Neighbour itna shor karta hai, neend hi nahi aati",
        "Ye project bohut boring hai, kuch samajh nahi aa raha",
        "Aaj kuch bhi theek nahi laga, bahut low feel ho raha hai",
        "Kal meeting hai office mein, prepare karna padega",
        "Train 10 minute late aa rahi hai platform number 3 se",
        "Delhi mein aajkal temperature 35 degree ke aaspaas hai",
        "Office ka timing 9 to 6 hai, Saturday off hota hai",
        "Petrol ki price phir se badh gayi hai is hafte",
        "Ye form online fill karna padega aur documents upload karne honge",
        "Match kal raat 8 baje start hoga Star Sports pe",
        "Is app mein payment UPI se bhi ho sakta hai",
        "College ke admission ke liye merit list Monday ko aayegi",
        "Ye medicine khane ke baad hi leni hai, before sleep",
        "Ghar se airport 45 minute ka drive hai normally",
        "Is course mein 3 modules hain, har module 2 weeks ka hai",
        "Aaj market band hai isliye bahar nahi jaunga",
        "Mera address sector 15, Noida hai near metro station",
        "Ye recipe mein dahi aur mirchi dono dalne padte hain",
    ],
    'sentiment': ['positive']*15 + ['negative']*15 + ['neutral']*15
}

df = pd.DataFrame(data)

# ── Stopwords ─────────────────────────────────────────────────────────────────
hindi_stops = {
    'hai', 'hain', 'tha', 'thi', 'mein', 'se', 'ka', 'ki', 'ke',
    'ko', 'par', 'pe', 'aur', 'toh', 'hi', 'bhi', 'jo', 'ye',
    'woh', 'hum', 'tum', 'aap', 'ab', 'kuch', 'koi', 'sab',
    'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'aaya', 'aayi',
    'mere', 'mera', 'meri', 'iska', 'unka', 'isliye'
}
english_stops = set(stopwords.words('english'))
all_stops = hindi_stops | english_stops

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in all_stops and len(t) > 1]
    return ' '.join(tokens)

# ── Language ID ───────────────────────────────────────────────────────────────
english_vocab = set(w.lower() for w in nltk_words.words())
hindi_vocab = {
    'yaar','bhai','dost','bahut','bohot','ekdum','bilkul','achha','theek',
    'aaj','kal','kya','nahi','haan','khana','ghar','dil','kaam','din',
    'raat','log','baat','khush','dukhi','mehnat','tension','poora','thoda',
    'phir','wala','wali','rang','didi','zyada','sirf','kaafi','pagal',
    'zabardast','kharab','bura','neend','shor','paisa','salary','promotion'
}

def tag_word(word):
    w = word.lower().strip('.,!?')
    if w in hindi_vocab:   return 'HI'
    if w in english_vocab and len(w) > 2: return 'EN'
    return 'UNK'

def lang_ratio(text):
    tags = [tag_word(w) for w in text.split()]
    c = Counter(tags)
    total = len(tags) or 1
    return round(c['HI']/total*100), round(c['EN']/total*100), round(c['UNK']/total*100)

# ── Train ─────────────────────────────────────────────────────────────────────
le = LabelEncoder()
X = df['text'].apply(clean).values
y = le.fit_transform(df['sentiment'].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

tfidf_w = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
tfidf_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1000)

X_tr = hstack([tfidf_w.fit_transform(X_train), tfidf_c.fit_transform(X_train)])
X_te = hstack([tfidf_w.transform(X_test),  tfidf_c.transform(X_test)])

lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
lr.fit(X_tr, y_train)
svm.fit(X_tr, y_train)

lr_acc  = round(accuracy_score(y_test, lr.predict(X_te))  * 100, 1)
svm_acc = round(accuracy_score(y_test, svm.predict(X_te)) * 100, 1)

print(f"✓ Models trained  —  LR: {lr_acc}%   SVM: {svm_acc}%")

# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    text = body.get("text", "").strip()
    model_choice = body.get("model", "lr")   # "lr" or "svm"

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean(text)
    feat = hstack([tfidf_w.transform([cleaned]), tfidf_c.transform([cleaned])])

    model = lr if model_choice == "lr" else svm
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]

    sentiment = le.inverse_transform([pred])[0]
    scores = {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}

    hi_pct, en_pct, unk_pct = lang_ratio(text)
    word_tags = [{"word": w, "lang": tag_word(w)} for w in text.split()]

    return jsonify({
        "sentiment":  sentiment,
        "confidence": round(float(max(proba)), 3),
        "scores":     scores,
        "word_tags":  word_tags,
        "lang_mix":   {"hindi": hi_pct, "english": en_pct, "unknown": unk_pct},
        "model_used": "Logistic Regression" if model_choice == "lr" else "SVM (Linear)"
    })

@app.route("/model_info")
def model_info():
    return jsonify({
        "lr_accuracy":  lr_acc,
        "svm_accuracy": svm_acc,
        "train_size":   len(X_train),
        "test_size":    len(X_test),
        "classes":      list(le.classes_),
        "features":     2500
    })

if __name__ == "__main__":
    app.run(debug=True)
