import pandas as pd
import numpy as np

#Finds all the positions where char is in word
def findall(char,word):
    pos=[word.find(char)]
    suma=pos[0]+1
    while pos[-1]!=-1:
        pos.append(word[suma:].find(char))
        suma+=pos[-1]+1#(hay que sumar 1 por el "_" que hay que borrar)
    if pos[0]==-1:
        return pos
    pos=pos[:-1]
    suma=pos[0]
    for i in range(1,len(pos)):
        suma+=pos[i]+1#Hay que sumar 1 por los arrays empiezan en 0)
        pos[i]=suma
    return pos


def De_OH_column_names(columns_df):
    """Tries to find all the columns of the original dataframe, os numerical+De_OH_encoded"""
    future_colnames=[]
    for i in range(len(columns_df)):
        pos=findall("_",columns_df[i])
        if pos==[-1] and columns_df[i] not in future_colnames:
            future_colnames.append(columns_df[i])
            continue
        
        names=[columns_df[i][:p] for p in pos]
        #We´re gonna go over all the possibilities, so each round we´ve got a longer name
        #We are gonna assume that the longest possible name is the correct name
        name_final=""
        for name in names:
            Has_a_match=False#possible_columns=[]
            long=len(name)
            for j in range(len(columns_df)):
                #We can´t check ourselves because we would just return our own OH encoded name
                if j==i:       
                    continue 
                if name==columns_df[j][:long]:
                    #If the column isn´t actually followed by a _ we don´t want it (we´re looking for other OH encoded like us, which if it exists it´ll be followed by a _)       
                    if len(columns_df[j])>long:       
                        if columns_df[j][long]!="_":       
                            continue  
                    Has_a_match=True#possible_columns.append(columns_df[j])
            if Has_a_match:#len(possible_columns)>=1:
                name_final=name
            #It´s possible that we´re alone as it´s a binary and drop first-> we take the longest name
        if name_final not in future_colnames and name_final!="":
            future_colnames.append(name_final)
        #It´s possible that we don´t find a match, and most likely it´s because we´re dealing with a numerical (not OH)
        if name_final=="" and columns_df[i] not in future_colnames:
            future_colnames.append(columns_df[i])

    return future_colnames


def DeOneHotEncodeColumn(df_true,colname,inplace=False):
    """This function decodes onehotencoded columns that start with colname (and eliminates those columns)
        if dropfirst=True, we won´t have all the info, so the restoration of those values will simply be "" which will have to be dealt by the user
    """
    if inplace:
        df=df_true.copy()
    else:
        df=df_true
    #These are the columns we think are onehotencoded
    cols_tryout=[col for col in df.columns if colname+"_"==col[:len(colname)+1]]
    #Maybe colname is the only onehotencoded column (due to dropfirst)->there won´t be any column longer as colname isn´t the root
    if cols_tryout==[]:
        cols_tryout=[colname]
    #The cols from cols_tryout aren´t necessarily binary, so we´ll check
    cols=[]
    for col in cols_tryout:
        EsBin = df[col].isin([0, 1]).all()

        if EsBin:
            cols.append(col)
    
    #If there aren´t any binary columns... what are u doing lol
    if len(cols)==0:
        return df
    #If there´s only one binary column, it´s easy (either it´s that column or it´s not (we add a not :)))
    elif len(cols)==1:
        col=cols[0]
        pos=findall("_",col)
        colname=col[:pos[-1]]
        sol=np.array([col[len(colname)+1:] if x else "not "+col[len(colname)+1:] for x in df[col]],dtype="object")
    else:
        sol=np.array(["" for i in range(df.shape[0])],dtype="object")
        #Let´s get the output of our decoded column ()
        for col in cols:
            sol+=np.array([col[len(colname)+1:] if x else "" for x in df[col]],dtype="object")

    #Let´s drop the columns we used and put the decoded column in place :)
    df[colname]=sol
    df=df.drop(cols,axis=1)
    return df

#We´re gonna detect the columns, sort their names so that EVENT_TYPE is done before EVENT...and done :)
def DeOneHotEncode(df_OH,exclude=[],final_cols=[]):
    """This is the function to use, it decodes all the onehotencoded columns in the dataframe
        We give the option of using final_cols to specify the columns we want to decode, and exclude to specify the columns we don´t want to decode
        final_cols removes the search we would normally do for decoding OH cols
    """
    df=df_OH.copy()
    if len(final_cols)>0:
        cols=pd.Series(final_cols)
    else:
        cols=pd.Series(De_OH_column_names(df.columns))
    cols=cols.sort_values(ascending=False)
    for col in cols:
        if col not in exclude:
            df=DeOneHotEncodeColumn(df,col,inplace=True)
    return df