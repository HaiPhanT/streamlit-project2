import streamlit as st
import pandas as pd
import numpy as np
import csv
import sys
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, dump
from surprise.model_selection import cross_validate, GridSearchCV

PAGES = {"HOME": "Home", "INSIGHT": "Insight", "COLLABORATIVE_FILTERING": "Collaborative Filtering",
    "CONTENT_BASED_FILTERING": "Content-Based Filtering", }

# Track cursor index

san_pham_cursor = 0


# Data handlers

def load_data():
    csv.field_size_limit(sys.maxsize)

    df_danh_gia = pd.read_csv('data/Danh_gia.csv')
    df_khach_hang = pd.read_csv('data/Khach_hang.csv')
    df_san_pham = pd.read_csv('data/San_pham.csv')

    return df_danh_gia, df_khach_hang, df_san_pham


# Recommendation systems

def get_recommendations_by_user(user_id, df_danh_gia, algo, num=10):
    df_score = df_danh_gia[["ma_san_pham"]]
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: algo.predict(user_id, x).est)
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()
    df_score = df_score[df_score['EstimateScore'] >= 3]

    return df_score.head(num)


def get_similar_recommendations(sp_id, cosine_sim, df, nums=3):
    idx = df.index[df['ma_san_pham'] == sp_id][0]
    print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]  # Lấy 3 sản phẩm tương tự nhất
    sp_indices = [i[0] for i in sim_scores]
    return df.iloc[sp_indices]


def get_same_price_recommendations(sp_id, df, price_threshold=500_000, nums=3):
    # Lấy thông tin sản phẩm đã chọn
    product = df[df['ma_san_pham'] == sp_id].iloc[0]

    # Lọc sản phẩm cùng phân khúc giá (cùng hoặc gần ngưỡng giá của sản phẩm đã chọn)
    price_range = (product['gia_ban'] - price_threshold, product['gia_ban'] + price_threshold)
    similar_price_products = df[(df['gia_ban'] >= price_range[0]) & (df['gia_ban'] <= price_range[1])]

    # Loại bỏ sản phẩm đã chọn khỏi danh sách đề xuất
    similar_price_products = similar_price_products[similar_price_products['ma_san_pham'] != sp_id]

    # Chọn 3 sản phẩm tương tự theo giá
    return similar_price_products.head(nums)


def get_same_usage_recommendations(sp_id, df, nums=3):
    # Lấy thông tin mô tả của sản phẩm đã chọn
    product = df[df['ma_san_pham'] == sp_id].iloc[0]

    # Tính toán độ tương đồng giữa các sản phẩm dựa trên mô tả sản phẩm
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['mo_ta'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Lấy chỉ số sản phẩm tương tự
    idx = df.index[df['ma_san_pham'] == sp_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sắp xếp sản phẩm theo độ tương đồng giảm dần
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]  # Bỏ sản phẩm chính

    # Lấy danh sách các sản phẩm tương tự
    sp_indices = [i[0] for i in sim_scores]
    return df.iloc[sp_indices]


def get_same_brand_recommendations(sp_id, df, brands_list, nums=3):
    # Lấy thông tin sản phẩm đã chọn
    product = df[df['ma_san_pham'] == sp_id].iloc[0]

    # Tìm thương hiệu của sản phẩm đã chọn trong cột 'mo_ta'
    brand = None
    for b in brands_list:
        if b.lower() in product['mo_ta'].lower():
            brand = b
            break

    # Nếu không tìm thấy thương hiệu trong mô tả sản phẩm, trả về thông báo
    if brand is None:
        print(f"Không tìm thấy thương hiệu trong mô tả sản phẩm {sp_id}.")
        return pd.DataFrame()

    # Lọc sản phẩm cùng thương hiệu (dựa vào mô tả)
    same_brand_products = df[df['mo_ta'].str.contains(brand, case=False, na=False)]

    # Loại bỏ sản phẩm đã chọn khỏi danh sách đề xuất
    same_brand_products = same_brand_products[same_brand_products['ma_san_pham'] != sp_id]

    # Đề xuất 'nums' sản phẩm cùng thương hiệu
    return same_brand_products.head(nums)


def get_recommendations(sp_id, cosine_sim, df, recommend_by='Similarity', nums=5):
    if recommend_by == "Price":
        return get_same_price_recommendations(sp_id, df, 500_000, nums)
    elif recommend_by == "Usage":
        return get_same_usage_recommendations(sp_id, df, nums)
    elif recommend_by == "Brand":
        brands_list = ["L'Oréal", 'Maybelline', 'Shiseido', 'Olay', 'Neutrogena', 'Clinique', 'Lancome', 'Klairs']
        return get_same_brand_recommendations(sp_id, df, brands_list, nums)
    else:
        return get_similar_recommendations(sp_id, cosine_sim, df, nums)


# GUI components

def build_drawable_sidebar():
    open_login = False

    st.sidebar.title("Select Recommendation System")

    page = st.sidebar.radio("", list(PAGES.values()))
    st.session_state.page = page

    if st.sidebar.button("Login"):
        open_login = True

    st.sidebar.info(f'Login as {st.session_state.user_name if "user_name" in st.session_state else "Guest"}')

    st.sidebar.title("Authors")
    st.sidebar.info("Trần Thị Kim Hoa")
    st.sidebar.info("Phan Thanh Hải", )

    return open_login


@st.dialog("Login")
def display_login_form(df_khach_hang):
    st.write("Select user to login with")
    user_names = df_khach_hang['ho_ten'].tolist()
    user_ids = df_khach_hang['ma_khach_hang'].tolist()
    user_dict = dict(zip(user_names, user_ids))

    search_query = st.selectbox("Search", user_names)

    if st.button("Submit"):
        st.session_state.user_id = user_dict.get(search_query)
        st.session_state.user_name = search_query
        st.rerun()


def display_recommendations_by_user(user_id, df_danh_gia, df_san_pham, algo, num=10):
    recommendations = get_recommendations_by_user(user_id, df_danh_gia, algo, num)

    st.write("Recommendations:")
    for index, row_data in recommendations.iterrows():
        san_pham = df_san_pham[df_san_pham['ma_san_pham'] == row_data['ma_san_pham']]
        with st.expander(str(san_pham['ten_san_pham'])):
            st.write(
                f"Description: {san_pham['mo_ta'][:300] + '...' if len(san_pham['mo_ta']) > 300 else san_pham['mo_ta']}")
            st.write(f"Price: {san_pham['gia_ban']}")
            st.write(f"Rating: {row_data['EstimateScore']}")


def display_list_san_pham(df_san_pham):
    global san_pham_cursor
    page_size_limit = 100

    number_of_san_pham = len(df_san_pham)
    item_per_row = 3
    page_size = 10 * item_per_row

    st.write("List of products")

    number_of_rows = math.ceil(number_of_san_pham / item_per_row)
    rows = [st.columns(3) for _ in range(int(number_of_rows))]

    for index, row_data in df_san_pham.iterrows():
        if index > page_size_limit:
            break

        col = rows[int(index / 3)][index % 3]
        with col:
            with st.popover(row_data['ten_san_pham']):
                st.write(f"Price: {row_data['gia_ban']}")
                st.write(
                    f"Description: {row_data['mo_ta'][:500] + '...' if len(row_data['mo_ta']) > 500 else row_data['mo_ta']}")
                st.write(f"Rating: {row_data['diem_trung_binh']}")
                st.write(f"Category: {row_data['phan_loai']}")

def display_insight(df):
    st.write("Welcome to the Insight page")
    st.write("This page is under construction")

def display_recommend_by():
    recommend_by = st.selectbox("Recommend by:", ["Similarity", "Price", "Usage", "Brand"])

    return recommend_by


def display_product(df_san_pham, product_id):
    product = df_san_pham[df_san_pham['ma_san_pham'] == product_id]
    st.write(product)


def display_recommendations(recommendations):
    for index, row_data in recommendations.iterrows():
        with st.expander(row_data['ten_san_pham']):
            st.write(f"Price: {row_data['gia_ban']}")
            st.write(
                f"Description: {row_data['mo_ta'][:300] + '...' if len(row_data['mo_ta']) > 300 else row_data['mo_ta']}")
            st.write(f"Rating: {row_data['diem_trung_binh']}")


def build_search_bar(df_san_pham):
    product_names = df_san_pham['ten_san_pham'].tolist()
    product_ids = df_san_pham['ma_san_pham'].tolist()
    product_dict = dict(zip(product_names, product_ids))

    search_query = st.selectbox("Search", product_names)

    return product_dict.get(search_query)

def home_page(df_san_pham):
    st.title("Home")
    st.write("Welcome to the Home page")
    display_list_san_pham(df_san_pham)

def insight_page():
    st.title("Insight")
    st.write("Welcome to the Insight page")
    st.write("This page is under construction")


def app():
    df_danh_gia, df_khach_hang, df_san_pham = load_data()
    cosine_model = np.load('models/cosine_sim.npy')
    _, svd_algo = dump.load('models/svd_model')

    # pg = st.navigation([st.Page(home_page(df_san_pham), 'home'), st.Page(insight_page(), 'insight')])
    # pg.run()

    if 'page' not in st.session_state:
        st.session_state.page = PAGES["HOME"]

    open_login = build_drawable_sidebar()

    if open_login:
        display_login_form(df_khach_hang)

    if st.session_state.page == PAGES["HOME"]:
        st.title("Home")
        st.write("Welcome to the Home page")
        display_list_san_pham(df_san_pham)
    elif st.session_state.page == PAGES["INSIGHT"]:
        st.title("Insight")
        display_insight(df_san_pham)
    elif st.session_state.page == PAGES["COLLABORATIVE_FILTERING"]:
        st.title("Collaborative Filtering")
        st.write("Welcome to the Collaborative Filtering page")

        if 'user_id' not in st.session_state or st.session_state.user_id is None:
            st.write("Please login to use Collaborative Filtering")
        else:
            display_recommendations_by_user(st.session_state.user_id, df_danh_gia, df_san_pham, svd_algo)
    elif st.session_state.page == PAGES["CONTENT_BASED_FILTERING"]:
        st.title("Content-Based Filtering")
        st.write("Welcome to the Content-Based Filtering page")
        product_id = build_search_bar(df_san_pham)
        recommend_by = display_recommend_by()

        if product_id:
            display_product(df_san_pham, product_id)

        st.write("Recommendations:")
        recommendations = get_recommendations(product_id, cosine_model, df_san_pham, recommend_by)
        display_recommendations(recommendations)


app()
