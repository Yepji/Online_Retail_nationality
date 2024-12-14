#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:11:47 2024

@author: yeji
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


#import data
df = pd.read_excel('/Users/yeji/Library/CloudStorage/OneDrive-성균관대학교/3-2 Tübingen/Data Science Laboratory with Python/Final_Project/Data/Online Retail.xlsx')


#get the information of dataframe
print('Information of Dataframe')
print(df.info())


quantity = df['Quantity'].unique()
print(len(quantity))


Desc = df['Description'].unique()
print(len(Desc))

Customer = df['CustomerID'].unique()
print('Customer length')
print(len(Customer))

cancel = df['InvoiceNo'].apply(lambda x: str(x).startswith('C')).sum()
print(cancel)

cancel_q = df[df['Quantity'] < 0][['InvoiceNo', 'Quantity']]
print(cancel_q)


quantity_min = df[df['InvoiceNo'].apply(lambda x: str(x).startswith('C'))][['InvoiceNo', 'Quantity']]
print(quantity_min)

#null값 찾기
print(df['Description'].isnull().sum())
print(df['CustomerID'].isnull().sum())



def convert_alphabet_to_number(value):
    alphabet_to_number = {
        'A': '00', 'B': '01', 'C': '02', 'D': '03', 'E': '04', 'F': '05', 'G': '06', 'H': '07', 'I': '08', 'J': '09',
        'K': '10', 'L': '11', 'M': '12', 'N': '13', 'O': '14', 'P': '15', 'R': '16', 'S': '17', 'T': '18',
        'U': '19', 'V': '20', 'W': '21', 'Y': '22', 'Z': '23', 'a': '24', 'b': '25', 'c': '26', 'd': '27',
        'e': '28', 'f': '29', 'g': '30', 'h': '31', 'i': '32', 'j': '33', 'k': '34', 'l': '35', 'm': '36', 'n': '37',
        'o': '38', 'p': '39'
    }
    
    # 문자열 슬라이싱을 사용하여 마지막 문자 추출
    last_char = value[-1]
    
    # 마지막 문자를 딕셔너리에서 찾아 숫자로 변환
    if last_char in alphabet_to_number:
        return value[:-1] + str(alphabet_to_number[last_char])
    else:
        return value


# 데이터프레임 열 변환
df['StockCode'] = df['StockCode'].astype(str)
df['StockCode'] = df['StockCode'].apply(convert_alphabet_to_number)
print(df)

df.loc[df['StockCode'] == 'POS17', 'StockCode'] = '00017'
df.loc[df['StockCode'] == 'POS18', 'StockCode'] = '00018'
print(df[df['StockCode'] == '15056B11'])
# df['StockCode'] = df['StockCode'].astype(float)

df_2 = df.dropna(subset=['Description', 'CustomerID'])



'''
print(df)

#drop C from cancellation InvoiceNo
df_2['InvoiceNo'] = df_2['InvoiceNo'].astype(str)
df_2 = df_2[~df_2['InvoiceNo'].str.startswith('C')]



plt.figure(figsize=(12, 8))
sns.heatmap(df_2[['Quantity', 'UnitPrice', 'CustomerID']].corr(), annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.show()
'''
