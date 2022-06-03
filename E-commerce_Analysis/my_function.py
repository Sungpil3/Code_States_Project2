
# 두 집단의 위치모수 검정
def two_sample_test(arg_x, arg_y, arg_data, alter, outlier):
    import numpy as np
    import scipy.stats as stats   
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(f"{arg_x}에 따른 {arg_y}의 평균차")
    if outlier > 0:
        IQR = arg_data[arg_y].quantile(.75) - arg_data[arg_y].quantile(.25) 
        step = 1.5 * IQR
        upper_inner_fence = arg_data[arg_y].quantile(.75) + step * outlier
        lower_inner_fence = arg_data[arg_y].quantile(.25) - step * outlier
        arg_data = arg_data[(arg_data[arg_y] < upper_inner_fence) & (arg_data[arg_y] > lower_inner_fence)]    
    else:
        pass    
    com_list = arg_data[arg_x].unique()
    if len(com_list) != 2:
        raise Exception("집단의 개수가 2개가 아닙니다.")
    array_like1 = arg_data[arg_data[arg_x] == com_list[0]][arg_y]
    array_like2 = arg_data[arg_data[arg_x] == com_list[1]][arg_y]
    p = sns.boxplot(x =arg_x, y = arg_y, data = arg_data)
    # p.set_xlabel(arg_x, fontsize = 30)
    # p.set_ylabel(arg_y, fontsize = 30)    
    plt.show()
    # 표본의 수가 30이하면
    if (len(array_like1) < 30) or (len(array_like2) < 30):
        # 정규성 검정의 귀무가설을 기각한다면(모집단의 분포가 정규분포가 아니라면)
        if (stats.shapiro(array_like1)[1] < 0.05) or (stats.shapiro(array_like2)[1] < 0.05):
            # 등분산이 아니라면
            if stats.levene(array_like1, array_like2)[1] < 0.05:
                raise Exception("표본의 수가 작고 정규성을 따르지 않으며 등분산도 아니므로 두 집단의 위치모수를 비교할 수 없다.")
            # 등분산이라면 wilcoxon rank sum test
            else:    
                print(f"사용한 가설검정 : wilcoxon rank sum test")
                p_value = stats.ranksums(array_like1, array_like2, alternative=alter)[1]
        # 모집단의 분포가 정규분포를 따른다면
        else :         
            # 등분산이 아니라면
            if stats.levene(array_like1, array_like2)[1] < 0.05:
                print(f"사용한 가설검정 : 독립표본 T 검정 이분산")
                p_value = stats.ttest_ind(array_like1, array_like2, equal_var = False, alternative=alter)[1] # 이분산
            # 등분산이라면 
            else:    
                print(f"사용한 가설검정 : 독립표본 T 검정 등분산")
                p_value = stats.ttest_ind(array_like1, array_like2, alternative=alter)[1] # 등분산
    # 표본의 크기가 30이상이면
    else :
        # 등분산이 아니라면
            if stats.levene(array_like1, array_like2)[1] < 0.05:
                print(f"사용한 가설검정 : 독립표본 T 검정 이분산")
                p_value = stats.ttest_ind(array_like1, array_like2, equal_var = False, alternative=alter)[1] # 이분산
            # 등분산이라면 
            else:    
                print(f"사용한 가설검정 : 독립표본 T 검정 등분산")
                p_value = stats.ttest_ind(array_like1, array_like2, alternative=alter)[1] # 등분산            
    print(f"유의확률 = {p_value}")
    return p_value
    

# 전처리 함수
def prepro(df):
    from category_encoders import OneHotEncoder
    import numpy as np
    import datetime
    import pandas as pd
    # 결측치는 missing 문자열로 대체
    df = df.fillna("missing")

    # event_type one hot encoding
    encoder = OneHotEncoder(cols = ["event_type"], use_cat_names = True)
    df = encoder.fit_transform(df)

    # event_time 시간형 변수로 전환하고 GMT+4 반영
    df['event_time'] = pd.to_datetime(df['event_time'], format="%Y-%m-%d %H:%M:%S UTC")
    df['event_time'] = df['event_time'] + datetime.timedelta(hours=4)    
    
    # event_time의 날자, 시간, 요일 변수생성
    df["month"] = df["event_time"].dt.month
    df["day"] = df["event_time"].dt.day
    df["hour"] = df["event_time"].dt.hour
    df["week_day"] = df["event_time"].dt.weekday

    # category_code 세분화  
    df["category_code_0"] = df["category_code"].apply(lambda x : x.split(".")[0])
    return df

# time_split 으로 리스트를 받아서 시간대별로 구분해주는 함수
# {1 : "morning", 2 : "afternoon", 3 : "evening", 0 : "bedtime"}
def to_hour_category(hour, time_split):
    if (hour >= time_split[0]) and (hour < time_split[1]):
        return 1
    elif (hour >= time_split[1]) and (hour < time_split[2]):
        return 2
    elif (hour >= time_split[2]) and (hour < time_split[3]):
        return 3
    else:
        return 0

# 시간에 따른 회귀분석
def scatter_plot_by_hour(target):
    import pandas as pd
    import matplotlib.pyplot as plt    
    df = pd.read_pickle("../../data/groupby_time_data.pkl")
    df = df[["hour", target]]
    plt.scatter(x = df["hour"], y = df[target])
    plt.title(target)
    plt.show()

def reg_scatter(target, start, end, df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = pd.read_pickle("../../data/groupby_time_data.pkl")
    df = df[["hour", target]]
    if start > end:
        df["hour"] = df["hour"].apply(lambda x : x + 24 if x <= end else x)
        end += 24
    df = df[(df["hour"] >= start) & (df["hour"] <= end)]   
    sns.regplot(x = "hour", y = target, data = df)
    plt.title(target)
    plt.show()

def reg_summary(target, start, end):
    import pandas as pd
    import statsmodels.api as sm
    df = pd.read_pickle("../../data/groupby_time_data.pkl")
    df = df[["hour", target]]
    if start > end:
        df["hour"] = df["hour"].apply(lambda x : x + 24 if x <= end else x)
        end += 24
    df = df[(df["hour"] >= start) & (df["hour"] <= end)]
    model = sm.OLS.from_formula(target + "~" + "hour", data = df)
    result = model.fit()
    print(result.summary())
    return result

def residual_qqplot(target, start, end):
    import pandas as pd
    import scipy.stats as stats
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    df = pd.read_pickle("../../data/groupby_time_data.pkl")
    df = df[["hour", target]]
    if start > end:
        df["hour"] = df["hour"].apply(lambda x : x + 24 if x <= end else x)
        end += 24
    df = df[(df["hour"] >= start) & (df["hour"] <= end)]
    model = sm.OLS.from_formula(target + "~" + "hour", data = df)
    result = model.fit()
    stats.probplot(result.resid, dist=stats.norm, plot=plt)
    plt.title(f"QQ-plot of resid {target}")
    plt.show()

def residual_scatter(target, start, end):
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    df = pd.read_pickle("../../data/groupby_time_data.pkl")
    df = df[["hour", target]]
    if start > end:
        df["hour"] = df["hour"].apply(lambda x : x + 24 if x <= end else x)
        end += 24
    df = df[(df["hour"] >= start) & (df["hour"] <= end)]
    model = sm.OLS.from_formula(target + "~" + "hour", data = df)
    result = model.fit()
    plt.scatter(result.predict(), result.resid_pearson)
    plt.title(f"scatter plot of resid {target}")
    plt.show()    

# 분산분석
def ANOVA(arg_x, arg_y, arg_data, mc_print, outlier):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import pingouin as pg
    import scikit_posthocs as sp
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multicomp import MultiComparison
    print(f"{arg_x}에 따른 {arg_y}의 평균차")
    if outlier > 0:
        IQR = arg_data[arg_y].quantile(.75) - arg_data[arg_y].quantile(.25) 
        step = 1.5 * IQR
        upper_inner_fence = arg_data[arg_y].quantile(.75) + step * outlier
        lower_inner_fence = arg_data[arg_y].quantile(.25) - step * outlier
        arg_data = arg_data[(arg_data[arg_y] < upper_inner_fence) & (arg_data[arg_y] > lower_inner_fence)]    
    else:
        pass    
    # 독립변수의 array
    list_of_factor = arg_data[arg_x].unique()
    # 독립변수의 수
    no_of_factor = len(list_of_factor)
    # 독립변수의 수가 3보다 작으면 에러 발생
    if no_of_factor < 3:
        raise Exception('factor의 개수가 3 보다 작습니다. 독립표본 T검정 혹은 단일표본 T 검정을 실시하세요') 
    # 각 독립변수마다의 값이 저장된 list
    data_by_factor = [arg_data[arg_data[arg_x]== i][arg_y] for i in list_of_factor]
    # plt.figure(figsize=(no_of_factor * 2,16))
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    p = sns.boxplot(x =arg_x, y = arg_y, data = arg_data)
    # p.set_xlabel(arg_x, fontsize = 30)
    # p.set_ylabel(arg_y, fontsize = 30)    
    plt.show()
    # 정규성 검정을 통과한 수
    no_of_normal = 0
    for factor_data in data_by_factor:
        if len(factor_data) < 30:    
            if stats.shapiro(factor_data)[1] < 0.05:
                break
            else:
                no_of_normal += 1
        else:
            no_of_normal += 1            
    if no_of_normal == no_of_factor:
        print("정규성 확인완료 등분산검정 실시")
        if stats.levene(*data_by_factor)[1] < 0.05:
            print("이분산이 나타난다 welch anova 실시")
            p_val = pg.welch_anova(dv = arg_y, between = arg_x, data = arg_data)["p-unc"][0]
            if p_val < 0.05:
                print(f"유의확률 = {p_val} 분산분석의 귀무가설 기각, 다중비교 실시")
                mc = MultiComparison(data = arg_data[arg_y], groups = arg_data[arg_x])
                if mc_print == 1:
                    print(mc.tukeyhsd())
            else:         
                print(f"유의확률 = {p_val}분산분석의 귀무가설 기각불가능, {arg_x}별로 평균의 차이가 없다")
        else:
            print("등분산 확인완료 분산분석 실시")    
            p_val = stats.f_oneway(*data_by_factor)[1]
            if p_val < 0.05:
                print(f"유의확률 = {p_val} 분산분석의 귀무가설 기각, 다중비교 실시")
                mc = MultiComparison(data = arg_data[arg_y], groups = arg_data[arg_x])
                if mc_print == 1:
                    print(mc.tukeyhsd())
            else:         
                print(f"유의확률 = {p_val}분산분석의 귀무가설 기각불가능, {arg_x}별로 평균의 차이가 없다")        
    else:
        print("정규성 확인실패 비모수적 분산분석인 kruskal wallis test으로 전환")
        p_val = stats.kruskal(*data_by_factor)[1]      
        if p_val < 0.05:
            print(f"유의확률 = {p_val} 분산분석의 귀무가설 기각, 다중비교 실시")
            if mc_print == 1:
                print(sp.posthoc_conover(arg_data, val_col = arg_y, group_col = arg_x, p_adjust = 'holm'))
            return sp.posthoc_conover(arg_data, val_col = arg_y, group_col = arg_x, p_adjust = 'holm')
        else:         
            print(f"유의확률 = {p_val}분산분석의 귀무가설 기각불가능, {arg_x}별로 평균의 차이가 없다")