# 필요모듈 불러오기
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import implicit
import pickle
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or RuntimeWarning)

# 로그데이터 학습하는 함수
def pickling_for_model(DATA_PATH):
    
    """
    로그데이터의 csv 파일의 위치를 받아서 추천 모델에 필요한 각종 객체를 pickling 하는 함수 
    DATA_PATH : 로그데이터의 csv 파일의 위치 ex) "../data/2019-Oct.csv"
    """
    
    def pickling(arg_object, arg_file_name):
        """
        arg_object를 현재위치에 arg_file_name.pkl 로 pickling 하는 함수 
        
        """
        with open(f'{arg_file_name}.pkl','wb') as pickle_file:
            pickle.dump(arg_object, pickle_file)       
        print(f"{arg_file_name}.pkl로 pickling 완료") 
    
    # 데이터를 column별로 불러올 수 있도록 data.parquet.gzip 으로 현재 위치에 저장
    # 사용하지 않는 event_time, category_id, user_session 은 제거한다.
    # 결측치는 제거하고 view 의 로그만 저장한다 
    # 대주제 카테고리인 category_code_0 변수도 만들어준다.
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace = True)
    df.drop(columns = ["event_time", "category_id", "user_session"], inplace = True)
    df = df[df["event_type"] == "view"]
    df.reset_index(drop = True, inplace = True)
    df["category_code_0"] = df["category_code"].apply(lambda x : x.split(".")[0])  
    df.to_parquet("view_data.parquet.gzip")
    del df
    
    # 로그 데이터에 해당 유저가 있는지 확인하기 위해 유저의 set을 users.pkl로 저장 
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["user_id"])
    users = set(df["user_id"].to_list())
    pickling(users, "users")
    del users, df

    # 전체 product_id중에서 상위 10개를 popular_product_id_list.pkl로 저장
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "event_type"])
    df = df.groupby("product_id").count()
    popular_product_id_list = list(df.sort_values("event_type", ascending=False).index[:10])
    pickling(popular_product_id_list, "popular_product_id_list")
    del popular_product_id_list, df

    # user_id를 key 로 가지고 해당 user가 view한 product의 개수를 value 로 가지는 dict를 user_unique_porduct_dict.pkl로 저장 
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["user_id", "product_id"])
    df = df.groupby("user_id").nunique()
    df = df.reset_index()
    user_unique_porduct_dict = {user_id : unique_product for user_id, unique_product in  list(zip(df["user_id"], df["product_id"]))}
    pickling(user_unique_porduct_dict, "user_unique_product_dict")
    del user_unique_porduct_dict, df

    # implicit를 이용한 als model 객체를 als_model.pkl로 저장 
    # 해당 모델에서 사용된 user와 product를 index로 변환하도록 
    # als_user_to_index.pkl, als_index_to_user.pkl, als_product_to_index.pkl, als_index_to_product.pkl저장
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["user_id", "product_id"])
    df = df.groupby("user_id").nunique()
    # 유저가 view한 product의 개수가 10보다 큰 유저들만 사용합니다.
    df = df[df["product_id"] > 10]
    upper_user_list = df.index
    del df
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["user_id", "product_id", "event_type"])
    df = df[df["user_id"].isin(upper_user_list)]  
    df = df.groupby(["user_id", "product_id"]).count().reset_index()
    del upper_user_list

    # 데이터를 csr_matrix로 만드는 과정입니다.
    user_unique = df['user_id'].unique()
    product_unique = df['product_id'].unique()
    als_user_to_index = {user:index for index, user in enumerate(user_unique)}
    als_index_to_user = {index:user for index, user in enumerate(user_unique)}
    als_product_to_index = {product:index for index, product in enumerate(product_unique)}
    als_index_to_product = {index:product for index, product in enumerate(product_unique)}
    
    # user와 product 의 index와의 변환을 위한 dict pickling
    pickling(als_user_to_index, "als_user_to_index")
    pickling(als_index_to_user, "als_index_to_user")
    pickling(als_product_to_index, "als_product_to_index")
    pickling(als_index_to_product, "als_index_to_product")
    df['user_id'] = df['user_id'].map(als_user_to_index.get)
    df['product_id'] = df['product_id'].map(als_product_to_index.get)
    num_user = df['user_id'].nunique()
    num_product = df['product_id'].nunique()
    user_item_matrix = csr_matrix((df.event_type, (df.user_id, df.product_id)), shape= (num_user, num_product))
    
    # user_item_matrix pickling
    pickling(user_item_matrix, "user_item_matrix")
    del als_user_to_index, als_index_to_user, als_product_to_index, als_index_to_product
    del df, user_unique, product_unique, num_user, num_product
    
    # 모델적합 
    alpha = 40
    als_model = implicit.als.AlternatingLeastSquares(factors=20, regularization=20, use_gpu=False, iterations=10, dtype=np.float32)
    als_model.fit((user_item_matrix * alpha).astype("double"))
    
    # als_model pickling
    pickling(als_model, "als_model")
    del als_model, user_item_matrix, alpha

    # 10개 이하의 product 를 view한 user_id를 받으면 가장 view가 많은 상품을 반환하는 dict 저장 (단 category_code나 brand가 결측값인 로그는 제거한다)
    # user_to_most_viewed_product_id

    df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "user_id"])
    df = df.groupby("user_id").nunique()
    df = df[df["product_id"] <= 10]
    lower_user_list = df.index
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["user_id", "product_id", "event_type"])
    df = df[df["user_id"].isin(lower_user_list)]  
    df = df.reset_index(drop =True)
    df = df.groupby(["user_id", "product_id"]).count().reset_index()    
    del lower_user_list

    # 데이터를 csr_matrix로 만드는 과정입니다.
    user_unique = df['user_id'].unique()
    product_unique = df['product_id'].unique()
    cb_user_to_index = {user:index for index, user in enumerate(user_unique)}
    cb_index_to_user = {index:user for index, user in enumerate(user_unique)}
    cb_product_to_index = {product:index for index, product in enumerate(product_unique)}
    cb_index_to_product = {index:product for index, product in enumerate(product_unique)}
    
    # user와 product 의 index와의 변환을 위한 dict pickling
    pickling(cb_user_to_index, "cb_user_to_index")
    pickling(cb_index_to_user, "cb_index_to_user")
    pickling(cb_product_to_index, "cb_product_to_index")
    pickling(cb_index_to_product, "cb_index_to_product")
    df['user_id'] = df['user_id'].map(cb_user_to_index.get)
    df['product_id'] = df['product_id'].map(cb_product_to_index.get)
    num_user = df['user_id'].nunique()
    num_product = df['product_id'].nunique()
    lower_user_item_matrix = csr_matrix((df.event_type, (df.user_id, df.product_id)), shape= (num_user, num_product))

    # user_item_matrix pickling
    pickling(lower_user_item_matrix, "lower_user_item_matrix")
    most_viewed_product_index = np.argmax(lower_user_item_matrix, axis = 1)
    most_viewed_product_index = np.squeeze(np.asarray(most_viewed_product_index))
    most_viewed_product_index = list(most_viewed_product_index)
    most_viewed_product_id = list(map(cb_index_to_product.get, most_viewed_product_index))
    del most_viewed_product_index

    lower_user_index = list(range(num_user))
    lower_user_id = list(map(cb_index_to_user.get, lower_user_index))
    del lower_user_index

    user_id_most_viewed_product_id = list(zip(lower_user_id, most_viewed_product_id))

    del most_viewed_product_id, lower_user_id
    user_to_most_viewed_product_id = {user_id : most_viewed_product_id for user_id, most_viewed_product_id in user_id_most_viewed_product_id }
    
    del user_id_most_viewed_product_id
    pickling(user_to_most_viewed_product_id, "user_to_most_viewed_product_id")
    
    del cb_user_to_index, cb_index_to_user, cb_product_to_index, cb_index_to_product
    del df, user_unique, product_unique, num_user, num_product
    
    # CB를 위한 Persontable을 category에 따라 pcikling
    def make_pearson_table(target_category):
        '''
        target_category 가 주어지면 해당 카테고리의 상품들로 pearson table을 만들어서 반환하는 함수
        '''

        # 데이터 불러오기 
        df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "category_code", "brand", "price", "category_code_0"])

        # 해당 카테고리만 가져오기
        df = df[df["category_code_0"] == target_category]
        df = df.reset_index(drop= True)

        # 카테고리와 브랜드를 합친 category_code+brand 변수 생성
        df["category_code+brand"] = df["category_code"] +  df["brand"].apply(lambda x : "." + x)

        # 제품별로 category_code+brand와 가격의 평균으로 보기
        df = df.groupby("product_id").agg({"category_code+brand" : "unique", "price" : "mean"})
        df = df.reset_index()
        df["category_code+brand"] = df["category_code+brand"].apply(lambda x : x[0])
        
        # 가격평균을 MinMaxScaler 를 이용하여 스케일링하기
        # df_minmax 는 스케일링된 가격평균을 가지고 있는 DataFrame
        scaler = MinMaxScaler()
        df_minmax = scaler.fit_transform(df[["price"]])
        df_minmax = pd.DataFrame(df_minmax, columns=['mMprice'])
        df_minmax.index = df['product_id'].values
        del scaler

        # CountVectorizer 적용
        # sparse matrix 인 countvect 에서 직접 계산하면 더 효율적일 것으로 예상
        vect = CountVectorizer()
        docs = df['category_code+brand'].values
        countvect = vect.fit_transform(docs)
        countvect_df =pd.DataFrame(countvect.toarray(), columns = sorted(vect.vocabulary_))
        countvect_df.index = df['product_id'].values
        del vect, docs, countvect

        # 제품을 index로 가지는 데이터(제품별 특징을 담고있다)
        df = pd.concat([df_minmax, countvect_df], axis= 1)
        del df_minmax, countvect_df

        # 피어슨 유사도 계산
        df = df.T.corr()
        return df    

    df = pd.read_parquet("view_data.parquet.gzip", columns = ["category_code_0"])
    category_code_list = list(df["category_code_0"].unique())
    del df

    # 각 카테고리별로 pearson table을 생성하고 저장한다.
    for category_code in category_code_list:
        df = make_pearson_table(target_category = category_code)
        pickling(df, f"{category_code}_pearson_table")
        del df        
    del category_code_list 

    # product_id 입력받으면 해당 제품의 1차 카테고리를 반환하는 dict 생성
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "category_code_0"])
    df = df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=True)
    product_id_to_category_code_0 = {product_id : category_code_0 for product_id, category_code_0 in list(zip(df["product_id"], df["category_code_0"]))}
    pickling(product_id_to_category_code_0, "product_id_to_category_code_0")
    del product_id_to_category_code_0, df    

    # product_id 별로 category와 brand, mean_of_price 를 가지는 DataFrame 생성후 저장
    df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "price"])
    mean = df[["product_id", "price"]].groupby("product_id").mean()
    del df

    df = pd.read_parquet("view_data.parquet.gzip", columns = ["product_id", "category_code", "brand"])
    df = df.drop_duplicates(subset=["product_id"], keep='first', inplace=False, ignore_index=True)
    product = df.set_index("product_id")
    del df

    explain = pd.concat([product, mean], axis =1)
    explain = explain.fillna("missing")
    pickling(explain, "explain")
    del explain

# 저장한 pkl 파일을 불러오는 함수
def test_pkl(name):
    test = None
    with open(f'{name}.pkl','rb') as pickle_file:
        test = pickle.load(pickle_file)
    return test

# 추천 리스트를 설명해주는 함수
def explain_products(product_id_list):
    explain = test_pkl("explain")
    for count, product_id in enumerate(product_id_list):
        exp = explain[explain.index == product_id]
        print(f"{count + 1}번째 추천 product_id = {product_id}, category_code = {exp['category_code'][product_id]}, brand = {exp['brand'][product_id]}, mean_of_price = {round(exp['price'][product_id], 2)}")    

# Class Bns
class Bns():
    
    def __init__(self):
        pass
    
    # fit 메소드로 결과 출력에 필요한 데이터를을 생성하고 저장합니다.
    def fit(self, DATA_PATH):
        pickling_for_model(DATA_PATH)
    
    # user_id를 받으면 아직 보지않은 product_id 10개를 list로 반환합니다.
    def recommend(self, user_id):

        # 해당 유저가 로그데이터 안에 있다면
        if user_id in test_pkl("users"):
            print(f"user_id = {user_id} 는 로그데이터 안에 있습니다.")

            # 해당 유저가 조회한 상품의 수가 11개 이상이면
            if test_pkl("user_unique_product_dict")[user_id] > 10:
                print(f"user_id = {user_id} 는 view 한 product의 개수가 {test_pkl('user_unique_product_dict')[user_id]}으로  10보다 큽니다. CF 기반의 추천을 사용합니다.")
                als_model = test_pkl("als_model")
                user_item_matrix = test_pkl("user_item_matrix")
                als_user_to_index = test_pkl("als_user_to_index")
                user_index = als_user_to_index[user_id]
                recommendations = list(als_model.recommend(user_index, user_item_matrix[user_index])[0])
                als_index_to_product = test_pkl("als_index_to_product")
                return list(map(als_index_to_product.get, recommendations))
            
            # 해당 유저가 조회한 상품이 10개 이하이면
            else :
                print(f"user_id = {user_id} 는 view 한 product의 개수가 {test_pkl('user_unique_product_dict')[user_id]}으로 10이하입니다. CB 기반의 추천을 사용합니다.") 
                most_viewed_product_id = test_pkl("user_to_most_viewed_product_id")[user_id]
                print(f"해당 유저가 가장 많이 본 product_id = {most_viewed_product_id}")
                exp = test_pkl('explain')
                print(f"해당 유저가 가장 많이 본 product의 category_code = {exp[exp.index == most_viewed_product_id]['category_code'][most_viewed_product_id]}")
                print(f"해당 유저가 가장 많이 본 product의 brand = {exp[exp.index == most_viewed_product_id]['brand'][most_viewed_product_id]}")
                print(f"해당 유저가 가장 많이 본 product의 mean of price = {exp[exp.index == most_viewed_product_id]['price'][most_viewed_product_id]}")                    
                user_index = test_pkl("cb_user_to_index")[user_id]
                viewed_product_index_list = list(np.where(test_pkl("lower_user_item_matrix")[user_index].toarray()[0] != 0)[0])
                viewed_product_id_list = list(map(test_pkl("cb_index_to_product").get, viewed_product_index_list ))
                pearson_table = test_pkl(f"{test_pkl('product_id_to_category_code_0')[most_viewed_product_id]}_pearson_table")
                pearson_table = pearson_table[~pearson_table.index.isin(viewed_product_id_list)]
                return list(pearson_table[most_viewed_product_id].sort_values(ascending=False).index[:10])
        
        #  로그에 없는 신규유저라면
        else:
            print("로그에 없는 신규유저입니다. 가장 view가 많은 product 10개를 추천합니다.") 
            return test_pkl("popular_product_id_list")        