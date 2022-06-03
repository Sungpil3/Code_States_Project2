# Basic Usage  
```python
import Bns  

model = Bns.Bns()

model.fit(DATA_PATH)

recommend = model.recommend(user_id)

explain_products(recommend)
```

DATA_PATH : 로그데이터의 위치 ex) "../data/2019-Oct.csv"

explain_products 를 통하여 추천받는 상품 10개의 정보를 확인할 수 있습니다.
