import streamlit as st
import pandas as pd
data = pd.read_csv("data_dantri_chungkhoan.csv")
pip install streamlit
pip install pyngrok
print(data.shape)
data.describe()

data.drop_duplicates(subset=["content"],inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)

# Đọc tệp vietnamesestopword.txt và lưu các từ dừng vào một danh sách
stopwords = []
with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

print(f'Number of stopwords: {len(stopwords)}')
print(f'First 10 stopwords: {stopwords[:10]}')

import nltk
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('wordnet')

import re
def clean_para(content):
    """Làm sạch văn bản bằng cách loại bỏ dấu câu, chuyển đổi thành chữ thường và loại bỏ stopwords."""
    # Kiểm tra nếu van_ban không phải là chuỗi, thì chuyển thành chuỗi
    if not isinstance(content, str):
        content = ' '.join(content)
    # Loại bỏ dấu câu
    content = re.sub(r'[^\w\s]', '', content)
    # Chuyển đổi thành chữ thường
    content = content.lower()
    # Loại bỏ stopwords
    tu_vung = content.split()
    tu_vung_sach = [tu for tu in tu_vung if tu not in stopwords]
    return ' '.join(tu_vung_sach)

data['content_cleaned'] = data['content'].apply(clean_para)
print(data[['content_cleaned']])

from sklearn.model_selection import train_test_split
# Chia dữ liệu thành tập huấn luyện, xác thực và tập kiểm tra 70-15-15
text_train, text_val_test = train_test_split(data, test_size=0.3, random_state=42)
text_val, text_test = train_test_split(text_val_test, test_size=0.5, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Xây dựng pipeline và tìm kiếm tham số tốt nhất
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=clean_para)),
    ('svd', TruncatedSVD())
])
param_grid = {
    'tfidf__max_df': [0.8, 0.9],
    'tfidf__min_df': [1, 2],
    'svd__n_components': [2, 5, 10, 20]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(text_train['content_cleaned'])

# Huấn luyện mô hình với tham số tốt nhất
best_params = grid_search.best_params_
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=best_params['tfidf__max_df'], min_df=best_params['tfidf__min_df'], tokenizer=clean_para)),
    ('svd', TruncatedSVD(n_components=best_params['svd__n_components']))
])
model.fit(text_train['content_cleaned'])

from sklearn.metrics.pairwise import cosine_similarity
def summarize_text(text, model, train_data):
    text_vector = model.transform([text])

    train_vectors = model.named_steps['tfidf'].transform(train_data['content_cleaned'])
    train_lsa = model.named_steps['svd'].transform(train_vectors)

    # Đảm bảo rằng text_lsa và train_lsa có cùng số lượng thành phần
    n_components = min(text_vector.shape[1], train_lsa.shape[1])
    text_lsa = text_vector[:, :n_components]
    train_lsa = train_lsa[:, :n_components]

    # Tính toán độ tương đồng cosine giữa các câu và văn bản gốc
    similarities = cosine_similarity(text_lsa, train_lsa)[0]

    # Xử lý trường hợp không có độ tương đồng
    sentences = nltk.sent_tokenize(text)
    if sentences and similarities.size > 0:
        top_sentence = similarities.argsort()[-1]
        # Trả về câu đầu tiên nếu chỉ mục không hợp lệ
        return sentences[top_sentence] if top_sentence < len(sentences) else sentences[0]
    else:
        return "No similar sentences found."
    
    # Sử dụng hàm summarize_text để tạo tóm tắt cho các đoạn văn trong tập dữ liệu kiểm định
text_val['summary'] = text_val['content_cleaned'].apply(lambda x: summarize_text(x, model, text_train))
print(text_val[['content_cleaned', 'summary']])

# Sử dụng hàm summarize_text để tạo tóm tắt cho các đoạn văn trong tập dữ liệu kiểm tra
text_test['summary'] = text_test['content_cleaned'].apply(lambda x: summarize_text(x, model, text_train))
print(text_test[['content_cleaned', 'summary']])

# !pip install rouge-score

# Đánh giá bằng thang đo ROUGE
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_summaries(true_summaries, generated_summaries):
    rouge1, rouge2, rougeL = 0, 0, 0
    n = len(true_summaries)
    for true, generated in zip(true_summaries, generated_summaries):
        scores = scorer.score(true, generated)
        rouge1 += scores['rouge1'].precision
        rouge2 += scores['rouge2'].precision
        rougeL += scores['rougeL'].precision
    return {
        'rouge1': rouge1 / n,
        'rouge2': rouge2 / n,
        'rougeL': rougeL / n
    }

# Tính điểm bằng dữ liệu đã tạo

true_summaries_val = text_val['content']
generated_summaries_val = text_val['summary']

evaluation_scores = evaluate_summaries(true_summaries_val, generated_summaries_val)
print(f"ROUGE Scores on Validation Set: {evaluation_scores}")


true_summaries_test = text_test['content']
generated_summaries_test = text_test['summary']

evaluation_scores_test = evaluate_summaries(true_summaries_test, generated_summaries_test)
print(f"ROUGE Scores on Test Set: {evaluation_scores_test}")

# Streamlit UI
st.title("Text Summarization App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['content_cleaned'] = data['content'].apply(clean_para)
    st.write(data.head())

    if st.button('Summarize'):
        data['summary'] = data['content'].apply(lambda x: summarize_text(x, model, text_train))
        st.write(data[['content', 'summary']])

