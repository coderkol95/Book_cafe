import pandas as pd
import numpy as np
import scipy as s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 500)
from pylab import cm
import warnings
warnings.filterwarnings('ignore')   
from pywaffle import Waffle 
from copy import deepcopy
from sklearn import feature_selection


def prepare():
    
    dataf = pd.read_csv(r"C:/Users/Anupam/Downloads/Books_with_coffee.csv")
    dataf.columns = ["Timestamp", "Format", "Beverage", "Music","Target", "Frequency","Connect","Variety"]
    dataf["VarCount"] = dataf.Variety.apply(lambda stri: len(stri.split(",")))   #Count of variety of books read by people
    format_count = dataf.groupby('Format')['Format'].count()
    return dataf, format_count

def plot1(dataf, format_count):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(22,8))
    sns.countplot(dataf.Format, palette="summer", ax=ax[0])
    ax[0].set_xlabel("Book format")
    ax[0].set_ylabel("Preferred choice of readers")
    sns.boxplot(dataf.Format, dataf.VarCount, palette='summer', ax=ax[1])
    ax[1].set_xlabel("Book format")
    ax[1].set_ylabel("No. of different genres read")
    plt.suptitle("Readership analysis", fontsize=20)
    ax[0].text(-0.05,format_count["Paperback"]+1, s=format_count["Paperback"])
    ax[0].text(0.95,format_count["Hard cover"]+1, s=format_count["Hard cover"])
    ax[0].text(1.95,format_count["Ebook"]+1, s=format_count["Ebook"])
    ax[0].text(2.95,format_count["Audiobook"]+1, s=format_count["Audiobook"])

def postprocess(dataf):

    dataf = dataf[(dataf.Format=="Paperback") | (dataf.Format=="Hard cover")]
    d = dataf[["Variety"]]                                                                      #Only the variety column
    d.Variety = d.Variety.apply(lambda s:s.strip(' ').split(","))                               #Picking up the varieties of genres   
    genres = []
    for i in d.Variety:
        for j in i:
            genres.append(j.strip(' '))                                                         #Storing all the genre varieties in genres    

    genres = pd.DataFrame(genres).groupby(0)[0].count()
    genres.drop(['In fiction- fantasy fiction to be exact. Adventure books. Enid Blyton( a little kiddish ik). I also would like to read all the Vedas one by one.','None other than those required for my coursework','Something else'],axis=0,inplace=True) #Dropping single ultra specific entrie(s)
    genres=genres.sort_values(0)                                                                #Sorting the genres dataframe

    dataf.Connect.replace({'Yes! no. Well maybe....':'May connect','Yes':'Want to connect','No':'Do not want to connect'},inplace=True)
    d = dataf.groupby('Connect')['Connect'].count()                                             #Grouping preferences of "Want to connect" attribute
    pphc = pd.DataFrame(dataf.Target)                                                           #Main column of interest
    pphc.Target.replace({"I wish I could read them without buying a lot of books":"Want to read without buying",
    "I want to build a library duh!":"Want to build a library"}, inplace=True)      
    pphc = pphc.groupby('Target')['Target'].count()                                             #Grouping by interest to read books without buying
    dataf.Beverage.replace({'No drink necessary':'I do not drink but I know things','None':'I do not drink but I know things','No drink necessary':'I do not drink but I know things','I drink but not with books':'I do not drink but I know things'},inplace=True)
    bv =pd.DataFrame(dataf.groupby('Beverage')['Beverage'].count())                             #Grouping by interest of beverage consumption
    bv.drop(['There is no connection between books and beverage'],axis=0,inplace=True)
    bv.columns=['Count']

    return dataf, genres, pphc, d, bv




def plot2(dataf, genres):
    fig,ax=plt.subplots(nrows=1, ncols=2,figsize=(20,8))
    genres.plot(kind='barh', color="dodgerblue", ax=ax[0])
    #sns.catplot(genres, ax=ax[0])
    plt.suptitle('Diversity among readers', fontsize=20)
    ax[0].set_ylabel('Genre')
    ax[0].set_xlabel('Genre read by different people')
    sns.lineplot(x="VarCount", y="Frequency",data=dataf, ax=ax[1],color='royalblue')
    ax[1].set_xlabel('No. of different genres read by a reader')
    ax[1].set_ylabel('No. of books read by the reader in the last six months')    
    #ax[0].axhline(genres.mean(), color='red')  
    #ax[0].text(40,genres.mean()+0.5,"Mean", fontsize=18) 


def plot3(pphc,d):
    plt.figure(figsize=(20,8))
    plt.pie(pphc, autopct='%2.1f%%',colors=['lavender','royalblue'], explode=[0.02,0.02], pctdistance=1.05,labels=pphc.index, labeldistance=1.1)
    plt.title("Interest among readers about a book cafe", fontsize=22)
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.pie(d, radius=0.85, autopct='%1.1f%%', explode=[0.02,0.02,0.02], pctdistance=0.7, labels=d.index, labeldistance=0.55, colors=['mintcream','palegreen','mediumspringgreen'])
    plt.tight_layout()
    plt.show()

 

def plot4(bv):
    # To plot the waffle Chart 
    fig = plt.figure(FigureClass = Waffle, rows = 3, values = bv. Count, labels = list(bv.index) , figsize=(20,8))
    plt.title('Beverage preference of readers',fontsize=20)

def plot5(dataf):
    da = dataf[dataf.Beverage=='I do not drink but I know things']
    x = pd.DataFrame(da.groupby('Connect')['Connect'].count())
    x.columns=['Count']
    fig = plt.figure(FigureClass = Waffle, rows = 1, values = x.Count, labels = list(x.index) , figsize=(20,4))
    plt.title('Interest to meet fellow book readers among people who do not drink while reading')     


def plot6(dmod):
    dmod.Beverage.replace({'I do not drink but I know things':'No drink required','None':'No drink required','No drink necessary':'No drink required','I drink but not with books':'No drink required','Depends upon mood and time of day':'No drink required','There is no connection between books and beverage':'No drink required'},inplace=True)
    dmod.Connect.replace({'Yes! no. Well maybe....':'Maybe'},inplace=True)
    dmod.Music.replace({'Yes, like my life':'Yes to music','Nope':'No to music'},inplace=True)
    dk = dmod[["Beverage","Connect","Music"]].groupby(['Connect','Music','Beverage'])[["Beverage"]].count()
    dk.columns=["Count"]
    dk.sort_values(["Connect","Music"], ascending=[False,True], inplace=True)
    dk = pd.concat([dk.iloc[0:7,:],dk.iloc[13:20,:]])
    return dk

def modprocess(datax):

    datax = datax[datax.Beverage!='There is no connection between books and beverage']
    datax.Beverage.replace({'I do not drink but I know things':'No drink required','None':'No drink required','No drink necessary':'No drink required','I drink but not with books':'No drink required','Depends upon mood and time of day':'No drink required'},inplace=True)
    datax.Connect.replace({'Yes':'Yes to connect','No':'No to connect','Yes! no. Well maybe....':'Open to connect'}, inplace=True)
    datax.Music.replace({'Nope':'No to music','Yes, like my life':'Yes to music'},inplace=True)
    datax[list(pd.DataFrame(datax["Format"].unique())[0].sort_values())] = pd.get_dummies(datax.Format)
    datax[list(pd.DataFrame(datax["Beverage"].unique())[0].sort_values())] = pd.get_dummies(datax.Beverage)
    datax[list(pd.DataFrame(datax["Music"].unique())[0].sort_values())] = pd.get_dummies(datax.Music)
    datax.Target.replace({"I wish I could read them without buying a lot of books":1,"I want to build a library duh!":0}, inplace=True)
    datax[list(pd.DataFrame(datax["Connect"].unique())[0].sort_values())] = pd.get_dummies(datax.Connect)
    datax.drop(['Format','Beverage','Music','Connect','Timestamp','Variety'], axis=1, inplace=True)
    datax.drop(['Alcohol','No to connect','Audiobook','Ebook'],axis=1, inplace=True)
    return datax

def plot7(dframe):
    x = dframe.drop(['Target'],axis=1)
    y = dframe.Target
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    fs = SelectKBest(f_classif, k="all")
    fs.fit(x,y)
    sc = pd.concat([pd.DataFrame(x.columns),pd.DataFrame(fs.scores_)], axis=1)
    sc.columns = ['Feature','Score']
    sc.sort_values('Score', inplace=True, ascending=False)
    plt.figure(figsize=(20,10))
    sns.barplot(sc.Score, sc.Feature, color='seagreen')
    plt.title('Feature importance among people wanting to read books without buying')


def get_palette(pal,n):
    a=[]
    cmap = cm.get_cmap(pal, n)    # PiYG

    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        a.append(matplotlib.colors.rgb2hex(rgba))
    return a    

   
