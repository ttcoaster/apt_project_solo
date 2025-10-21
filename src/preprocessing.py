import datetime
import os

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def data_load(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df['is_train'] = 1
    test_df['is_train'] = 0

    print(f"훈련 데이터 : {train_df['계약년월'].min()} ~  {train_df['계약년월'].max()} ({len(train_df)}건)")
    print(f"테스트 데이터 : {test_df['계약년월'].min()} ~  {test_df['계약년월'].max()} ({len(test_df)}건)")

    all_df = pd.concat([train_df, test_df], axis=0)

    return all_df


def get_gu_group(gu_str):
    gu_group01 = ['강북구', '강서구', '관악구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '서대문구', '성북구', '은평구', '중랑구']
    gu_group02 = ['강동구', '광진구', '동작구', '마포구', '성동구', '양천구', '영등포구', '종로구', '중구']
    gu_group03 = ['강남구', '서초구', '송파구', '용산구']

    gu_group = None
    if gu_str in gu_group01: gu_group = "gu01"
    elif gu_str in gu_group02: gu_group = "gu02"
    elif gu_str in gu_group03: gu_group = "gu03"
    else: gu_group = None

    return gu_group

def transform_property_features(df):
    # 평방미터를 평으로 변환 (한국에서 주로 사용)
    df['size_pyeong'] = df['전용면적(㎡)'] / 3.3058
    df['size_pyeong'] = df['size_pyeong'].astype(int)

    # 크기 구간화
    size_bins = [0, 60, 85, 110, 135, 200, np.inf]
    size_labels = ['초소형', '소형', '중소형', '중형', '중대형', '대형']
    df['size_category'] = pd.cut(df['전용면적(㎡)'], bins=size_bins, labels=size_labels)
    # 면적 로그 변환
    df['log_size'] = np.log1p(df['전용면적(㎡)'])
    
    # 저/중/고층 분류
    df['floor_category'] = pd.cut(
        df['층'], 
        bins=[-np.inf, 1, 5, 15, np.inf], 
        labels=['지하/1층', '저층', '중층', '고층']
        )
    
    # 건물 연식 관련 변환
    df['contract_year'] = df['yyyymm_str'].str[:4]
    df['contract_year'] = df['contract_year'].astype(int)
    df['apt_age'] = df['contract_year'] - df['건축년도']
    
    # 연식 구간화
    age_bins = [0, 5, 10, 15, 20, 30, np.inf]
    age_labels = ['신축', '준신축', '중축', '중고', '구축', '노후', '정보없음']
    df['age_category'] = pd.cut(df['apt_age'], bins=age_bins, labels=age_labels[:-1])
    # Fill missing age_category values with '정보없음'
    df['age_category'] = df['age_category'].cat.add_categories('정보없음').fillna('정보없음')
    
    # 재개발 가능성 연령대 (30년 이상 1, 미만 0)
    df['redevelopment_candidate'] = np.where(df['apt_age'] >= 30, 1, 0)
    # Convert redevelopment_candidate to categorical
    df['redevelopment_candidate'] = df['redevelopment_candidate'].map({1: '재건축', 0: '일반'})
    
    df['gu'] = df['시군구'].str.split().str[1]
    df['dong'] = df['시군구'].str.split().str[2]

    df['gu_group'] = df['gu'].apply(get_gu_group)
    return df

def create_housing_market_features(df, apt_price_file):

    # 아파트 매매 지표 데이터 로드 
    apt_price_index = pd.read_pickle(apt_price_file)

    # Create mapping between apt_price_index and all_df
    price_index_mapping = {}
    for yyyymm in apt_price_index['yyyymm'].unique():
        for gu in apt_price_index.columns[:-1]:  # Exclude yyyymm column
            price_index_mapping[(yyyymm, gu)] = apt_price_index.loc[apt_price_index['yyyymm']==yyyymm, gu].values[0]

    # Map price index values to all_df
    df['apt_price_index'] = df.apply(lambda x: price_index_mapping.get((x['yyyymm_str'], x['gu']), None), axis=1)
    
    return df


def handle_missing_values(df, var_type):
    if var_type == 'category':
        # 범주형 변수 찾기
        cat_columns = [col for col in df.columns if df[col].dtype in ('object', 'category')]
        missing_columns = [col for col in cat_columns if df[col].isnull().any()]
        print(f'missing category columns : {missing_columns}')
    elif var_type == 'numeric':
        numeric_columns = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
        missing_columns = [col for col in numeric_columns if df[col].isnull().any()]
        print(f'missing numeric columns : {missing_columns}')
    if len(missing_columns) == 0:
        return []
    
    temp =df[missing_columns].isnull().sum()
    temp_df = pd.DataFrame(temp)

    temp_df.index.name = f'{var_type} column'
    temp_df.columns = ['count']
    temp_df['missing_ratio'] = temp_df['count'] / len(df)
    temp_df.sort_values(by='missing_ratio', ascending=False, ignore_index=True)
    
    high_missing_cols = temp_df[temp_df['missing_ratio'] > 0.7].index.tolist()

    return high_missing_cols


def create_economic_features(df, homeLoanRate_file):
#homeLoanRate
    
    homeLoanRate = pd.read_pickle(homeLoanRate_file)
    df = pd.merge(df, homeLoanRate, left_on='yyyymm_str', right_on='yyyymm', how='left')
    
    # 금리 변화 구간화
    df['interest_rate_trend'] = pd.cut(
        df['interest_rate_6m_change'], 
        bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf], 
        labels=['급격한하락', '하락', '안정', '상승', '급격한상승']
    )
    
    return df


def make_prepr_df(df):
    """전처리된 데이터프레임 생성"""
    prepr_df = df.copy()
    prepr_df.drop(columns=['시군구', '번지', '본번', '부번', '아파트명', '전용면적(㎡)', '계약일', '건축년도', '도로명', '등기신청일자', '거래유형', '중개사소재지', 'yyyymm_str', 'yyyymm'], inplace=True)
    
    return prepr_df

if __name__ == "__main__":

    apt_loc_file_path = 'data/apt_loc_info_250511.pkl'
    apt_price_file_path = 'data/apt_price_index.pkl'
    homeLoanRate_file_path = 'data/homeLoanRate_20250511.pkl'

    # Check if required pickle files exist
    required_files = [
        apt_loc_file_path,
        apt_price_file_path,
        homeLoanRate_file_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file_path} not found")
        


    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    all_df = data_load(train_file, test_file)

    apt_loc_info = pd.read_pickle(apt_loc_file_path)

    # Merge location_df with all_df based on '시군구', '번지', '아파트명'
    all_df = pd.merge(all_df, apt_loc_info[['시군구', '번지', '아파트명', 'apt_id', 'near_station', 'bus_dist', 'subway_dist', 'elementary_dist']], 
                    on=['시군구', '번지', '아파트명'],
                    how='left')
    all_df['yyyymm_str'] = all_df['계약년월'].astype(str)
    all_df['bus_dist'] = all_df['bus_dist'].fillna(-999)
    all_df['subway_dist'] = all_df['subway_dist'].fillna(-999)
    all_df['elementary_dist'] = all_df['elementary_dist'].fillna(-999)


    #아파트 특성 파생변수 생성
    all_df = transform_property_features(all_df)

    #아파트 매매 지표 파생변수 생성
    all_df = create_housing_market_features(all_df, apt_price_file_path)

    # 경제 지표 파생변수 생성
    all_df = create_economic_features(all_df, homeLoanRate_file_path)

    # 결측치 처리
    high_missing_cols = handle_missing_values(all_df, 'category')
    all_df.drop(columns=high_missing_cols, inplace=True)

    high_missing_cols = handle_missing_values(all_df, 'numeric')
    all_df.drop(columns=high_missing_cols, inplace=True)

    all_df.rename(columns={'좌표X_y': '좌표X', '좌표Y_y': '좌표Y', '계약년월':'contract_ym', '층':'floor'}, inplace=True)
    # target 값에 로그변환 적용 (log1p는 log(1+x)를 계산하여 0값도 처리 가능)
    all_df['log_target'] = np.log1p(all_df['target'])

    # Get current date in yyyymmdd format
    current_date = datetime.datetime.now().strftime('%y%m%d')
    output_filename = f'data/all_df_{current_date}.pkl'

    all_df.to_pickle(output_filename)

    prepr_df = make_prepr_df(all_df)

    prepr_df.to_pickle(f'data/prepr_df_{current_date}.pkl')
    print(f'prepr_df_{current_date}.pkl 파일 저장 완료')


