#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:12:02 2025

@author: anyuchen
"""



#讀取匯入資料

#print(df["ICUstayingdays"].count())
#df['ICUstayingdays']=df['ICUstayingdays']-1
#df['admission_day']=df['admission_day']-1
#print(df['ICUstayingdays'].count())
#刪除住院一天即死亡的,['ICUstayingdays']==1尋找符合天數為1的，df[df['ICUstayingdays']==1].index再獲取這些行的索引
#df.drop(df[df['ICUstayingdays']==0].index,inplace=True)
#print(df['ICUstayingdays'].count())
#df.drop(df[df['ICUstayingdays']==1].index,inplace=True)   #等同於df=df.loc[df['ICUstayingdays] != 1]
#print(df['ICUstayingdays'].count())
#將會使用到的特徵若為中文名，改成英文年齡改為age
df.rename(columns={'年齡':'age'},inplace=True)
df.rename(columns={'性別':'Gender'},inplace=True)
df.rename(columns={'分數':'EuroSCORE'},inplace=True)
#將血費的特徵欄位名稱名稱改為CV
df.rename(columns={'血費':'CV'},inplace=True)
print(df['ICUstaydaysnew'].count())

#將數值做轉換,將符合條件的設定不同的結果
import numpy as np
#將性別的male改為1,Female改為0
df['Gender']=np.where(df['Gender']=='M',1,0)

#將所有資料轉為數值型態
for feature in df.columns:
    df[feature]=pd.to_numeric(df[feature],errors='coerce')
#將CPBtime>150設為1,0-150設為0
df['CPBtime']=np.where(df['CPBtime'] < 150,0,1)
#將Cardiacarresttime>90設為1,0-90設為0
df['Cardiacarresttime']=np.where(df['Cardiacarresttime'] < 90,0,1)
#將CV > 12032設為1,<12032設為0
df['CV']=np.where(df['CV']>12032,1,0)
#新增兩個欄位，將ICU:0-14天為0，14天以上為1；ICU:0-8天為0，8天以上為1
new_columns=pd.DataFrame({
    'ICUstayingdays_14':np.where(df['ICUstaydaysnew'] <= 14,0,1),
    'ICUstayingdays_8':np.where(df['ICUstaydaysnew'] <= 8,0,1)
    })

#因為發現合併後資料筆數可能因為索引值有差異而變多(原本兩者均為591,合併完變642)，因此需要重置索引在合併
#df.reset_index(drop=True,inplace=True)
#new_columns.reset_index(drop=True,inplace=True)
#並將新欄位合併到df1中
df1=pd.concat([df,new_columns],axis=1)
#檢查所有資料數值是否有nan值,會得到bypassgraftnumber有36筆nan值
print(df1.isnull().sum())
#bypassgraftnumber為連續型資料，要選擇哪個填補方式較好？中位數？均值？眾數？回歸/機器學習填補?
#要先觀察bypassgraftnumber的資料型態及分佈：為數值型態,因此可以使用平均值取代，且會得到沒有缺失值
import seaborn as sns
sns.histplot(df1['bypassgraftnumber'],kde=True,bins=20)
plt.xlabel('bypassgraftnumber')
plt.ylabel('count')
plt.title('Distribution of bypassgraftnumber')
plt.show()
#會發現bypassgraftnumber的資料呈現偏態,且多集中在0 or 3,因此會建議使用中位數填補,若想要正準確可以用KNN or 回歸模型來填補
#若使用中位數填補
#df1['bypassgraftnumber']=df1['bypassgraftnumber'].fillna(df1['bypassgraftnumber'].median())
#若使用回歸or KNN填補：如果 bypassgraftnumber 與其他變數（如年齡、病情嚴重度）高度相關，可以考慮 回歸預測填補 或 KNN（K 近鄰）填補
from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=5)
df1['bypassgraftnumber']=imputer.fit_transform(df1[['bypassgraftnumber']])
print(df1['bypassgraftnumber'].isnull().sum())

#只保留需要的欄位，並將其分類成不同大類
patient_related_factors=df1.loc[:,["age","Gender","renaldysfunction","PAOD","mobility","redo","COPD","SBE","critical","DM_insulin"],]
caidiac_related_factors=df1.loc[:,["NYHA","ANGINA","LVdysfunction","OldMI","PAH"]]
operation_related_factors=df1.loc[:,["Emergency","NoOFProcedure","aorta_OP"]]
intro_operative_factors=df1.loc[:,["CPBtime","bypassgraftnumber","Cardiacarresttime",'CV']]

post_operative_factors=df1.loc[:,["PostOpARF_SCORE","PostOpMediastinitis_SCORE","PostOpCVA_SCORE","RespirotoryFailure_SCORE","PostopCompication_SCORE"]]
EuroScore=df1["EuroSCORE"]
outcome=df1.loc[:,["Ventilatordays","ICUstayingdays_14","ICUstayingdays_8","admission_day"]]

#資料的敘述性統計
print((df1["ICUstayingdays_14"]==1).sum())
print((df1["ICUstayingdays_14"]==0).sum())
print((df1["ICUstayingdays_8"]==1).sum())
print((df1["ICUstayingdays_8"]==0).sum())

#將所有需要的資料合併
df2=pd.concat([patient_related_factors,caidiac_related_factors,operation_related_factors,intro_operative_factors,post_operative_factors,outcome,EuroScore],axis=1)
#檢查所有欄位是否有缺失值,會得到每個欄位都沒有缺失值
print(df2.isna().sum().head(36))
#將資料分為X,y
X=pd.concat([patient_related_factors,caidiac_related_factors,operation_related_factors,intro_operative_factors],axis=1)
y_14=outcome['ICUstayingdays_14']
y_8=outcome['ICUstayingdays_8']

#數值型態資料：age,bypassgraftnumber,支持呼吸的天數Ventilatordays,admission_day住院超期天數
num_data=pd.concat([df2.age,df2.Ventilatordays,df2.bypassgraftnumber,df2.admission_day],axis=1)
#類別型資料：
category_data=X.drop(['age','bypassgraftnumber'],axis=1)
#做基本的統計分析：常態分佈/t檢定/卡方檢定/ANOVA
#數值型資料：age,Ventilatordays,bypassgraftnumber,admission_day，其餘皆為類別型資料
#常態分佈：只能針對數值型資料做常態分佈,資料量>5000使用kstest(),資料量<5000使用shapiro
# from scipy.stats import kstest
# for feature in num_data.columns:
#     stat,p=kstest(num_data[feature],'norm',args=(num_data[feature].mean(),num_data[feature].std()))
#     #print(f"Feature:{feature} stat:{stat:.4f} p:{p:.4f}")
#     if p >= 0.05:
#         print(f"{feature}符合常態分佈")
#     else:
#         print(f"{feature}不符合常態分佈")
# #可視化所有特徵的常態分佈:長條圖
# import matplotlib.pyplot as plt
# import seaborn as sns
# for feature in X.columns:
#     sns.histplot(num_data[feature],kde=True,stat='count',bins=30)
#     plt.title(f"Distribution of {feature}")
#     plt.xlabel(f"Value of {feature}")
#     plt.ylabel('Count')
#     #plt.savefig(f"{feature}_distribution.png") #並將所有圖形存下來
#     plt.show()
#     plt.close()
#Shapiro-Wilk 檢定常態分佈：先觀察所有特徵的資料分佈,原先用kstest檢定(假設輸入是連續變數,故效果不佳),改用Shapiro-Wilk 檢定
from scipy.stats import shapiro
for feature in num_data.columns:
    stat,p=shapiro(num_data[feature].dropna())  #dropna()是為了確保沒有nan值
    #print(f"Feature:{feature} stat:{stat:.4f} p:{p:.4f}")
    if p >= 0.05:
        print(f"{feature}符合常態分佈")
    else:
        print(f"{feature}不符合常態分佈")
#可視化所有特徵的常態分佈:長條圖
import matplotlib.pyplot as plt
import seaborn as sns
for feature in num_data.columns:
    sns.histplot(num_data[feature],kde=True,stat='count',bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(f"Value of {feature}")
    plt.ylabel('Count')
    #plt.savefig(f"{feature}_distribution.png") #並將所有圖形存下來
    plt.show()
    plt.close()
#會得到所有特徵都不符合常態分佈

#自變數與因變數之間的顯著性關係
#獨立樣本t檢定(T-test(數值型Xvs類別型y-2類))：針對數值型的自變數與因變數之間是否具有顯著性之關係,但他需要樣本為常態分佈，但本研究的資料特徵為非常態分佈，因此要使用Mann-Whitney U檢定
# from scipy.stats import ttest_ind
# for feature in num_data.columns:
#     group1=num_data[df2["ICUstayingdays_14"]==1][feature]
#     group0=num_data[df2["ICUstayingdays_14"]==0][feature]
#     stat,p=ttest_ind(group1,group0,equal_var=False)
#     print(f"{feature}的t檢定結果：t={stat:.4f},p={p:.4f}")
#     if p < 0.05:
#         print(f"{feature}在ICU住院天數 >= 14天 與 < 14天有顯著性差異")
#mannwhitneyu檢定(數值型自變數vs類別型因變數)，來衡量變數與目標變數ICUstayingdays_14,ICUstayingdays_8的關係，且因資料為不常態分佈，應使用Mann-Whitney U檢定
from scipy.stats import mannwhitneyu
def manu_test(target,features):
    manu_results=[]
    for feature in features.columns:
        u_stat,p_value=mannwhitneyu(features[feature],target)
        result={
            'Feature':feature,
            'manu_stat':u_stat,
            'P-Value':p_value,
            'Significance':'顯著性關係' if p_value <= 0.05 else '不顯著關係'}
        manu_results.append(result)
        print(f"Festure:{feature} 與目標變數的關係為 {'顯著性關係' if p_value <= 0.05 else '不顯著關係'} (P-value:{p:.4f})")
    return pd.DataFrame(manu_results)
print("\n------ICUstayingdays_14的t檢定------")
manu_test_14=manu_test(y_14,num_data)
print("\n------ICUstayingdays_8的t檢定------")
manu_test_8=manu_test(y_8,num_data)

#卡方檢定(多元類別型vs類別型變數)：觀察兩變數之間是否有顯著性關係，各特徵與ICUstayingdays_14之間的關係，p-value <= 0.05為顯著性關係
chi2_data=category_data
print(chi2_data.dtypes) #可以得到所有的特徵欄位均為數值型態int64
from scipy.stats import chi2_contingency
#新增一個Cramer'sV計算函數，避免過度依賴P值
def cramers_v(confusion_matrix):
    chi2=chi2_contingency(confusion_matrix)[0]
    n=confusion_matrix.sum().sum()
    min_dim=min(confusion_matrix.shape)-1
    return np.sqrt(chi2/(n*min_dim))    

#將卡方檢定用一個函數包裝起來，那卡方檢定最終需要輸入的資料為輸入變數及輸出變數
def chi2_test(target,features):
    """
    target:目標變數Ｉ=ICUstayingdays_14,ICUstayingdays_8,需為一維資料
    features:所有考量的特徵資料，包含多個特徵
    """
    chi2_results=[]
    for feature in features.columns:
        #建立列連表,需確認輸入數據為一維
        contingency_table=pd.crosstab(target,features[feature])
        #執行卡方檢定
        chi2,p,dof,expected=chi2_contingency(contingency_table) 
        #計算Cramer's V
        cramers_v_value=cramers_v(contingency_table)
        #輸出結果為
        result={
            'Feature':feature,
            'Chi2_stat':chi2,
            'P-value':p,
            'Cramers_V':cramers_v_value,
            'Significance':'顯著性關係' if p <= 0.05 else '不顯著關係'}
        chi2_results.append(result)
        #顯示結果
        print(f"Feature:{feature} 與 目標變數間的關係為{'顯著性關係' if p <= 0.05 else '不顯著關係'} (P-value:{p:.4f}) | Cramer's V:{cramers_v_value:.4f} | {result['Significance']}")
        #print("\n---------ICUstayingdays_14卡方檢定的列連表----------\n")
        print(contingency_table)
    return pd.DataFrame(chi2_results) 
    
#使用所有特徵資料與ICUstayingdays_14,ICUstayingdays_8做檢定
print("\n---------對ICUstayingdays_14做卡方檢定----------\n")
chi2_results_14=chi2_test(y_14,chi2_data)

print("\n---------對ICUstayingdays_8做卡方檢定----------\n")
chi2_results_8=chi2_test(y_8,chi2_data)

# chi2_result=[]
# for feature_14 in chi2_data.columns:
#     #建立卡方檢定需要的列連表,需確認輸入數據為一維
#     contingency_table_14=pd.crosstab(y_14,chi2_data[feature_14])
#     #執行卡方檢定
#     chi2,p,dof,expected=chi2_contingency(contingency_table_14)
#     print(f"Feature:{feature_14} chi2_stat:{chi2:.4f} p-value:{p:.4f}")
#     if p <= 0.05:
#         print(f"Feature:{feature_14}與ICUstayingdays_14之間為顯著性關係")
#     else:
#         print(f"Feature:{feature_14}與ICUstayingdays_14之間為不顯著性關係")
# for feature_8 in chi2_data.columns:
#     #建立卡方檢定需要的列連表,需確認輸入數據為一維
#     contingency_table_8=pd.crosstab(y_8,chi2_data[feature_8])
#     #執行卡方檢定
#     chi2,p,dof,expected=chi2_contingency(contingency_table_8)
#     print(f"Feature:{feature_8} chi2_stat:{chi2:.4f} p-value:{p:.4f}")
#     if p <= 0.05:
#         print(f"Feature:{feature_8}與ICUstayingdays_8之間為顯著性關係")
#     else:
#         print(f"Feature:{feature_8}與ICUstayingdays_8之間為不顯著性關係")

#部分類別型資料具有不同程度的差異0/1/2/3...，因此要將其各自分開獨立跑回歸，看不同程度對y的影響
#包含renaldysfunction,NYHA,LVdysfunction,PAH,Emergency,NoOFProcedure
print(category_data.columns)
multi_category_data=category_data[['renaldysfunction','NYHA','LVdysfunction','PAH','Emergency','NoOFProcedure']]
#輸出各特徵欄位的值，並分類
#print(multi_category_data['NoOFProcedure'].unique())
#將邏輯回歸寫成一個函式
import statsmodels.api as sm 
def logis_reg(x,y,label=""):
    #要確保x是數值型，要對類別進行One-Hot Encoding
    #x=pd.get_dummies(multi_category_data[x],drop_first=True,prefix=x).astype(int)
    #加入常數項
    x=sm.add_constant(x)
    #建立邏輯回歸模型
    model=sm.Logit(y,x)
    result=model.fit()
    #計算Odds Ratio
    odds_ratios=np.exp(result.params)
    conf=np.exp(result.conf_int())
    or_df=pd.DataFrame({"OR":odds_ratios,"2.5%":conf[0],"97.5%":conf[1]})
    #輸出結果
    print(f"\n-----{label} 的邏輯回歸結果-----")
    print(result.summary())
    print("\n Odds Ratios:\n")
    print(or_df)
    return result
#先將multi_category_data的資料一起pd.get_dummies
#先確認變數均為類別變數
print(multi_category_data.info())
#先將資料轉為類別
multi_category_data=multi_category_data.astype('category')
print(multi_category_data.info())
multi_category_data=pd.get_dummies(multi_category_data,drop_first=True).astype(int)
# | 運算符表示邏輯 OR（如果 Emergency_2 或 Emergency_3 任一為 1，則新變數 Emergency_2 為 1）
multi_category_data['Emergency_2']=multi_category_data['Emergency_2']|multi_category_data['Emergency_3']
print(multi_category_data.columns)
#renaldysfunction:0/1/2/3
logis_reg(multi_category_data[['renaldysfunction_1','renaldysfunction_2','renaldysfunction_3']],y_14,label="renaldysfunction 對 ICU14天 影響")
logis_reg(multi_category_data[['renaldysfunction_1','renaldysfunction_2','renaldysfunction_3']],y_8,label="renaldysfunction 對 ICU14天 影響")
#NYHA:0/1/2/3
logis_reg(multi_category_data[['NYHA_1','NYHA_2','NYHA_3']],y_14,label="NYHA 對 ICU14天 影響")
logis_reg(multi_category_data[['NYHA_1','NYHA_2','NYHA_3']],y_8,label="NYHA 對 ICU8天 影響")
#LVdysfunction:0/1/2/3
logis_reg(multi_category_data[['LVdysfunction_1','LVdysfunction_2','LVdysfunction_3']],y_14,label="LVdysfunction 對 ICU8天 影響")
logis_reg(multi_category_data[['LVdysfunction_1','LVdysfunction_2','LVdysfunction_3']],y_8,label="LVdysfunction 對 ICU8天 影響")
#PAH:0/1/2
logis_reg(multi_category_data[['PAH_1', 'PAH_2']],y_14,label="PAH 對 ICU14天 影響")
logis_reg(multi_category_data[['PAH_1', 'PAH_2']],y_8,label="PAH 對 ICU8天 影響")
#Emergency:0/1/2/3,因為3的樣本數過少，無法跑回歸，所以把3跟2合併在2的欄位
logis_reg(multi_category_data[['Emergency_1','Emergency_2']],y_14,label="Emergency 對 ICU14天 影響")
logis_reg(multi_category_data[['Emergency_1','Emergency_2']],y_8,label="Emergency 對 ICU8天 影響")
#因為Emergency中的3類樣本數極少，會造成Singular matrix的錯誤，因此把3併進去2(或刪掉)，只保留0/1/2
# print(X['Emergency'].unique())
sns.catplot(x=X['Emergency'],hue=y_8,kind='count',palette='coolwarm')
plt.show()
# print(multi_category_data.groupby('Emergency')['y_14'].values_counts())
# #Emergency似乎與y_8具有高度共線性，因此計算Emergency_data的VIF
# #計算相關矩陣
# corr_matrix=(multi_category_data[['Emergency_1', 'Emergency_2','Emergency_3']]).corr()
# sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f')
# plt.show()
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# #計算VIF
# Eme=sm.add_constant(multi_category_data[['Emergency_1', 'Emergency_2','Emergency_3']])
# vif_data=pd.DataFrame({
#     "feature":Eme.columns,
#     "VIF":[variance_inflation_factor(Eme.values,i) for i in range(Eme.shape[1])]})
# print("\n VIF of Emergency:")
# print(vif_data)
#NoOFProcedure:0/1/2/3
logis_reg(multi_category_data[['NoOFProcedure_1', 'NoOFProcedure_2', 'NoOFProcedure_3']],y_14,label="NoOFProcedure 對 ICU14天 影響")
logis_reg(multi_category_data[['NoOFProcedure_1', 'NoOFProcedure_2', 'NoOFProcedure_3']],y_8,label="NoOFProcedure 對 ICU8天 影響")

#數值資料：num_data
#類別型資料：category_data
#所有x特徵一個一個丟下去跑
print(num_data.columns)
from sklearn.preprocessing import StandardScaler
#需要先數值變數標準化
scaler=StandardScaler()
#需要先數值age','bypassgraftnumber'變數標準化
X[['age','bypassgraftnumber']]=scaler.fit_transform(X[['age','bypassgraftnumber']])
#再將其他特徵欄位資料型態轉為類別
#將類別變數one hot encoding，one hot encoding是針對沒有次序性的多類別欄位
logis_reg(X,y_14,label="所有特徵 對 ICU14day 影響")
logis_reg(X,y_8,label="所有特徵 對 ICU8day 影響")

#變數之間是否有共線性問題？VIF
#輸入的x為數值型資料(標準化後的數值變數,one hot encoding後的類別變數)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(x):
    vif_data=pd.DataFrame()
    vif_data["Feature"]=x.columns   #變數名稱
    vif_data['VIF']=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
    print(f"=== Variance Inflation Factor ===")
    print(vif_data)
#計算VIF所有特徵X之間的貢獻性
calculate_vif(X)

print(y_14.isnull().sum())
print(y_8.isnull().sum())
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#svm=SVC(kernel='linear',random_state=24,class_weight='balanced')
#svm.fit(X_resample,y_resample)
#y_pred=svm.predict(X_test)
#class_report_svm=classification_report(y_test,y_pred)
#print(class_report_svm)

#為了確保每個模型使用的訓練測試集都相同，因此要先將資料平衡，然後切割完，再帶入函數中
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote=SMOTE(random_state=24)
X_resample,y_resample=smote.fit_resample(X,y_14)
#同步確保EuroSCORE擴增
euro_resample,_=smote.fit_resample(df1[['EuroSCORE']],y_14)
print("過採樣的數據資料：",y_resample.value_counts())
X_train,X_test,y_train,y_test,euro_train,euro_test=train_test_split(X_resample,y_resample,euro_resample,test_size=0.2,random_state=24)
#先將資料把EuroSCORE刪除，因為與ICUstayingdays共線性過高
#X_train_model=X_train.drop("EuroSCORE",axis=1)
#X_test_model=X_test.drop("EuroSCORE",axis=1)
#確認一下Euro的比數
print("過採樣的數據資料：",euro_resample.info())

#利用各種模型來訓練，要使用平衡後的資料X_resample,y_resample
#def輸入的參數會決定函數如何運行與返回的結果
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#因為已經透過GridSearch自動訓練尋找最佳參數，不需要手動訓練，因此可以不用再手動呼叫函數時輸入X_train,X_test,y_train,y_test
def model_train(model,X_train,X_test,y_train,y_test):
    """
    訓練模型並評估
    返回:訓練好的模型及測試集預測結果,預測概率
    """

    #訓練模型
    model.fit(X_train,y_train)
    #模型預測
    y_pred=model.predict(X_test)
    print(f"======{model}模型評估:=====")
    print("準確率(Accuracy):",accuracy_score(y_test,y_pred))
    print("分類報告(Classification Report):\n",classification_report(y_test,y_pred))
    #預測概率
    #y_prob = None 的目的是為了初始化變數 y_prob，並在後續的邏輯中根據不同情況來決定其具體的值,避免在未能計算出預測概率的情況下返回一個未賦值的變數。
    #當我們訓練分類模型並希望獲得每個樣本的預測結果時，我們常常會遇到兩種常見情況：一種是直接獲取每個樣本屬於某個類別的概率，另一種是獲取每個樣本距離決策邊界的距離。
    y_prob=None
    if hasattr(model,"predict_proba"):
        y_prob=model.predict_proba(X_test)[:,1]  #僅針對二元分類返回類別1的概率
    elif hasattr(model,"decision_function"):
        y_prob=model.decision_function(X_test)  #獲取樣本到SVM邊界的距離,正的數值表示樣本位於類別1的一側，負的數值表示樣本位於類別0的一側
    #因為後續需要繪製ROC曲線，因此需要回傳X_test,y_test
    return model,y_pred,y_prob

#定義繪製多個ROC curve函數，ROC curve需要參數：y_test,y_prob,fpr,tpr,auc_score
from sklearn.metrics import roc_auc_score,roc_curve
from itertools import cycle   #用於生成色彩循環
def plot_roc_curve(results,y_test):   #接收一個包含每個模型及其預測概率的字典 results 和真實標籤 y_test。   
    #使用matplotlib的色彩循環
    colors=cycle(["blue","green",'red','purple','orange','brown'])
    #results 是一個字典，存儲了每個模型（例如 RandomForest、SVM、KNN）的名稱以及對應的預測結果。results.items() 是一個內建方法，用來返回字典中的所有鍵值對。每個鍵值對都包含：
    #鍵（key）：即模型的名稱（例如 "Random Forest"、"SVM" 等）。
    #值（value）：一個二元組 (model, y_prob)，其中：model 是訓練好的模型,y_prob 是對應的預測概率（即模型對測試集的預測結果，特別是屬於正類的概率）。
    #name 是模型的名稱，而 (model, y_prob) 是模型和預測概率這兩個元素的二元組
    for name,(model,y_prob) in results.items():
        if y_prob is not None:
            #繪製ROC curve
            fpr,tpr,thresholds=roc_curve(y_test,y_prob)
            auc_score=roc_auc_score(y_test,y_prob)
            #繪製ROC曲線
            color=next(colors)
            plt.plot(fpr,tpr,color=color,label=f"{name}-ROC Curve (AUC={auc_score:.4f})")
        else:
            print(f"{name}不支持概率預測，略過繪製")
    #添加基線(可有可無)
    plt.plot([0,1],[0,1],color='gray',linestyle='--',label="Random Base Line")
    plt.title("All ROC Curve",fontsize=16)
    plt.xlabel('False Positive Rate(FPR)',fontsize=14)
    plt.ylabel('True Positive Rate(TPR)',fontsize=14)
    plt.legend(loc='lower right',fontsize=12) 
    plt.grid()
    plt.savefig("/Users/anyuchen/Desktop/ROC curve.jpg",dpi=500)
    plt.tight_layout()
    plt.show()

#定義模型集參數空間
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import cycle  #用於生成色彩循環
from sklearn.model_selection import GridSearchCV
models_param_grids={
        "Random Forest":{
            "model":RandomForestClassifier(),
            "params":{
                "n_estimators":[10,50,100],
                "criterion":["gini",'entropy'],
                "max_depth":[None,10,20,30]
                }
            },
        "SVM":{
            "model":SVC(probability=True),
            "params":{
                "kernel":['linear','rbf'],
                "C":[0.1,1,10],
                "gamma":['scale','auto']
            }
        },
        "KNN":{
            "model":KNeighborsClassifier(),
            "params":{
                "n_neighbors":[3,5,7],
                "weights":['uniform','distance'],
                "p":[1,2]
            }
        },
        "Logistic Regression":{
        "model":LogisticRegression(),
        "params":{
            "C":[0.1],
            "penalty":['l2'],
            "max_iter":[1000]
            }
        }
    }
#需要將結果儲存在results中，因為繪製ROC只需要y_prob的結果，因此需要將所有模型的結果儲存在一起，在繪圖時一次叫出來即可
results={}
for name,config in models_param_grids.items():
    print(f"Performing GridSearchCV for {name}...")
    grid_search=GridSearchCV(estimator=config["model"],
                             param_grid=config["params"],
                             cv=5,
                             scoring='roc_auc',
                             verbose=2,
                             n_jobs=1)
    grid_search.fit(X_train,y_train)
    #取得最佳模型
    best_model=grid_search.best_estimator_
    y_prob=best_model.predict_proba(X_test)[:,1]  #獲取正類別概率
    #呼叫model_train來做評估，因為model_train()函式回傳return model,y_pred,yprob,但因為我的程式碼中只需要y_pred,y_prob,因此填了一個_為忽略
    
    _,y_pred,y_prob=model_train(best_model,X_train,X_test,y_train,y_test)
    
    results[name]=(best_model,y_prob)
    
    print(f"Best parameters for {name} : {grid_search.best_params_}")
    #trained_model,y_pred,y_prob=model_train(model, X_train, X_test, y_train, y_test)
    #results[name]=(trained_model,y_prob)
#先新增y_euro_test,y_euro_prob到ROC曲線圖中,並因為EuroSCORE為百分比，因此需要除以100
y_euro_prob=euro_test['EuroSCORE']/100
fpr_euro,tpr_euro,thresholds=roc_curve(y_test,y_euro_prob)
euro_auc_score=roc_auc_score(y_test,y_euro_prob)
#將euro_roc_curve加到result中
results["EuroSCORE"]=("EuroSCORE Model",y_euro_prob)
#繪製ROC曲線
plot_roc_curve(results,y_test)
print(results)
#繪製隨機森林特徵重要性
if "Random Forest" in results and hasattr(results["Random Forest"][0],"feature_importances_"):  #提取訓練好的隨機森林模型
    #提取訓練好的隨機森林模型
    rf_model=results["Random Forest"][0]  
    feature_importance=rf_model.feature_importances_
    print("Feature Importance:\n",feature_importance)
    #X_rf_imp=X.drop("EuroSCORE",axis=1)
    #繪製特徵重要性圖表
    plt.bar(X.columns,feature_importance,color='steelblue',align='center')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance of Randomforest')
    plt.xticks(rotation=45,ha='right')  #特徵名稱選轉，避險重疊
    plt.tight_layout()  #自通調整子圖參數，防止標籤溢出
    plt.savefig("/Users/anyuchen/Desktop/bar_chart_feature_importance.jpg",dpi=500)
    plt.show()

#SHAP Analysis 並繪製Beeswarm plot
#因RandomForest模型效果最好，以它來做分析，最佳參數為Random Forest : {'criterion': 'entropy', 'max_depth': None, 'n_estimators': 50}
import shap
#從results中獲取訓練好的隨機森林模型,[0]即為訓練後的模型
rf_model=results['Random Forest'][0]
#創建SHAP解釋氣，對於隨機森林可以使用TreeExplainer
explainer=shap.TreeExplainer(rf_model)
#取得SHAP值
shap_values=explainer.shap_values(X_test)
print(f"shap_values:{shap_values}")
#檢查SHAP值的形狀
print(f"X_test shape:{X_test.shape}")
print(f"shap_values[1] shape:{shap_values[1].shape}")
#因為是目標變數是二分類問題，shap_values 是一個包含兩個元素的列表：
#其中shap_values[0]是負類（類別0）的 SHAP 值，shap_values[1] 是正類（類別1）的 
shap_values_class_1=shap_values[:,:,1]  #只關心正類
#繪製Beeswarmplot
shap.summary_plot(shap_values_class_1,X_test)
plt.show()
plt.savefig("/Users/anyuchen/Desktop/beeswarm_plot.png",dpi=500)


#針對沒有SMOTE資料,建立沒有平衡的資料才能與平衡後做比較,自變數：X,因變數：y_14,y_8
from sklearn.model_selection import train_test_split
X_train_0,X_test_0,y_train_0,y_test_0,euro_train_0,euro_test_0=train_test_split(X,y_14,df1[['EuroSCORE']],random_state=24,test_size=0.2)
#利用不同的模型做訓練:SVM,RandomForest,Logistic,KNN)
def model_train_0(model,X_train,X_test,y_train,y_test):
    """
    訓練模型並評估
    返回:訓練好的模型及測試集預測結果,預測概率
    """
    #訓練模型
    model.fit(X_train,y_train)
    #模型預測
    y_pred_0=model.predict(X_test)
    print(f"==={model}預測評估結果===")
    print("準確率(Accuracy):",accuracy_score(y_test,y_pred_0))
    print("分類報告(Classification Report):\n",classification_report(y_test,y_pred_0))
    #預測概率
    #y_prob = None 的目的是為了初始化變數 y_prob，並在後續的邏輯中根據不同情況來決定其具體的值,避免在未能計算出預測概率的情況下返回一個未賦值的變數。
    #當我們訓練分類模型並希望獲得每個樣本的預測結果時，我們常常會遇到兩種常見情況：一種是直接獲取每個樣本屬於某個類別的概率，另一種是獲取每個樣本距離決策邊界的距離。
    y_prob_0=None
    #hasattr(object, attribute)用來**檢查某個物件是否有指定的屬性
    if hasattr(model,"predict_proba"):
        #如果模型支持predict_proba(ex:邏輯回歸,隨機森林)
        y_prob_0=model.predict_proba(X_test)[:,1]  #取類別1的機率
    elif hasattr(model,"decision_function"):
        y_prob_0=model.decision_function(X_test)  #獲取樣本到SVM邊界的距離,正的數值表示樣本位於類別1的一側，負的數值表示樣本位於類別0的一側
    #因為後續需要繪製ROC曲線，因此需要回傳X_test,y_test
    return model,y_pred_0,y_prob_0

#針對沒有SMOTE資料，繪製多個ROC Curve,需要的參數：y_test,y_prob,fpr,tpr,auc_score
def plot_roc_curve_0(results_0,y_test):  #接收一個包含每個模型及其預測概率的字典 results 和真實標籤 y_test。
    #使用matplotlib的色彩循環
    colors=cycle(['blue','green','red','purple','orange','brown'])   
    #results是一個字典，包含了每個薄型(RandomForest,SVM,KNN)的名稱以及其對應的結果。results.items() 是一個內建方法，用來返回字典中的所有鍵值對。每個鍵值對都包含：
    #鍵（key）：即模型的名稱（例如 "Random Forest"、"SVM" 等）。
    #值（value）：一個二元組 (model, y_prob)，其中：model 是訓練好的模型,y_prob 是對應的預測概率（即模型對測試集的預測結果，特別是屬於正類的概率）。
    #name 是模型的名稱，而 (model, y_prob) 是模型和預測概率這兩個元素的二元組
    for name,(model,y_prob_0) in results_0.items():
        if y_prob_0 is not None:
            #繪製ROC Curve
            fpr,tpr,thresholds=roc_curve(y_test_0,y_prob_0)
            auc_score=roc_auc_score(y_test_0,y_prob_0)
            color=next(colors)
            plt.plot(fpr,tpr,color=color,label=f"{name}-ROC Curve (AUC={auc_score:.4f})")
        else:
            print(f"{name}不支持概率預測，忽略繪圖")
    #添加基線(可有可無)
    plt.plot([0,1],[0,1],color='gray',linestyle='--',label='Random Base Line')
    plt.title("All ROC Curve",fontsize=16)
    plt.xlabel('False Positive Rate(FPR)',fontsize=14)
    plt.ylabel('True Positive Rate(TPR)',fontsize=14)
    plt.legend(loc='lower right',fontsize=12)
    plt.grid()
    plt.savefig("/Users/anyuchen/Desktop/ROC curve.jpg",dpi=500)
    plt.tight_layout()
    plt.show()

#針對沒有SMOTE資料，定義模型集的參數空間
models_param_grids_0={
    "Random Forest":{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[10,50,100],
            'criterion':['gini','entropy'],
            'max_depth':[None,10,20,30]
            }
        },
    "SVM":{
        'model':SVC(probability=True),
        'params':{
            'kernel':['linear','rbf'],
            'C':[0.1,1,10],
            'gamma':['scale','auto']
            }
        },
    "KNN":{
        'model':KNeighborsClassifier(),
        'params':{
            'n_neighbors':[3,5,7],
            'weights':['uniform','distance'],
            'p':[1,2]
            }
        },
    "Logistic Regresstion":{
        'model':LogisticRegression(),
        'params':{
            'C':[0.1],
            'penalty':['l2'],
            'max_iter':[1000]
            }
        }
    }

#針對沒有SMOTE資料，需要將結果除存在results中,因為繪製ROC只需要y_prob的結果，因此需要將所有模型的結果儲存在一起，在繪圖時一次叫出來即可
results_0={}
for name,config in models_param_grids_0.items():
    print(f"Performing GridSearchCV for {name}...")
    grid_search_0=GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='roc_auc',
        verbose=2,
        n_jobs=1)
    grid_search_0.fit(X_train_0,y_train_0)
    best_model_0=grid_search_0.best_estimator_
    y_prob_0=best_model_0.predict_proba(X_test_0)[:,1]
    results_0[name]=(best_model_0,y_prob_0)
    print(f"Best parameter for {name} : {grid_search_0.best_params_}")
#新增y_euro_test,y_euro_prob到ROC曲線中，且因為EuroSCORE為百分比，故需要除以100
y_euro_prob_0=euro_test_0['EuroSCORE']/100
fpr_euro_0,tpr_euro_0,thresholds_0=roc_curve(y_test_0,y_euro_prob_0)
euro_auc_score_0=roc_auc_score(y_test_0,y_euro_prob_0)
#將euro_roc_curve加到result中
results_0["EuroSCORE"]=("EuroSCORE Model",y_euro_prob_0)
#繪製ROC曲線
plot_roc_curve(results_0,y_test_0)
print(results_0)

#針對沒有SMOTE資料繪製隨機森林特徵重要性
if "Random Forest" in results_0 and hasattr(results_0["Random Forest"][0],"feature_importances_"):  #提取訓練好的隨機森林模型
    #提取訓練好的隨機森林模型
    rf_model_0=results_0["Random Forest"][0]  
    feature_importance_0=rf_model_0.feature_importances_
    print("Feature Importance:\n",feature_importance_0)
    #X_rf_imp=X.drop("EuroSCORE",axis=1)
    #繪製特徵重要性圖表
    plt.bar(X.columns,feature_importance_0,color='steelblue',align='center')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance of Randomforest')
    plt.xticks(rotation=45,ha='right')  #特徵名稱選轉，避險重疊
    plt.tight_layout()  #自通調整子圖參數，防止標籤溢出
    plt.savefig("/Users/anyuchen/Desktop/bar_chart_feature_importance.jpg",dpi=500)
    plt.show()

#SHAP Analysis 並繪製Beeswarm plot
#因RandomForest模型效果最好，以它來做分析，最佳參數為Random Forest : {'criterion': 'entropy', 'max_depth': None, 'n_estimators': 50}
import shap
#從results中獲取訓練好的隨機森林模型,[0]即為訓練後的模型
rf_model_0=results_0['Random Forest'][0]
#創建SHAP解釋氣器，對於隨機森林可以使用TreeExplainer
explainer_0=shap.TreeExplainer(rf_model_0)
#取得SHAP值
shap_values_0=explainer_0.shap_values(X_test_0)
print(f"shap_values:{shap_values_0}")
#檢查SHAP值的形狀
print(f"X_test shape:{X_test_0.shape}")
print(f"shap_values[1] shape:{shap_values[1].shape}")
#因為是目標變數是二分類問題，shap_values 是一個包含兩個元素的列表：
#其中shap_values[0]是負類（類別0）的 SHAP 值，shap_values[1] 是正類（類別1）的 
shap_values_class_1=shap_values[:,:,1]  #只關心正類
#繪製Beeswarmplot
shap.summary_plot(shap_values_class_1,X_test_0)
plt.savefig("/Users/anyuchen/Desktop/beeswarm_plot.png",dpi=500)
plt.show()















