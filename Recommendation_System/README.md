# 추천 모델링

### 유저가 view를 실시한 상품이 10개 이하면 CB 11개 이상이면 CF
```flow
st=>start: user_id 입력
op1=>operation: 조회수가 가장높은 상품 10개 추천
cond1=>condition: 데이터에 user_id의 view로그가 있는가?
e=>end

st->cond
cond(yes)->e
cond(no)->op
```

```mermaid  
graph LR
A[Hard edge] -->B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
​```