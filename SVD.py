# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:34:10 2016

@author: Administrator
"""

import numpy as np
from numpy.random import random 
train_data = np.loadtxt('traindata.csv', dtype = object, delimiter = ',')
test_data = np.loadtxt('testdata.csv', dtype = object, delimiter = ',')
class  Userbased:  
    def __init__(self,X):  
        ''''' 
            k  is the length of vector 
        '''  
        
        self.X=np.array(X)  
        self.ave=np.mean(self.X[:,2].astype(float)) 
        print "the input data size is ",self.X.shape  
        self.movie_user={}  
        self.user_movie={}  
        for i in range(self.X.shape[0]):  
            uid=self.X[i][0]  
            mid=self.X[i][1]  
            rat=float(self.X[i][2])  
            self.movie_user.setdefault(mid,{})  
            self.user_movie.setdefault(uid,{})  
            self.movie_user[mid][uid]=rat  
            self.user_movie[uid][mid]=rat 
        self.similarity={} 
    def cal_sim(self,u1,u2): 
        self.similarity.setdefault(u1,{})
        self.similarity.setdefault(u2,{})
        self.similarity[u1].setdefault(u2,-1.0)
        self.similarity[u2].setdefault(u1,-1.0)
        if self.similarity[u1][u2]!=-1:
            return self.similarity[u1][u2]
        si = []
        siml = -2
        for item in self.user_movie[u1]:
            if item in self.user_movie[u2]:
                    si.append(item)
        if len(si)!=0:
            user1 = []
            user2 = []
            for i in si:
                user1.append(self.user_movie[u1][i])
                user2.append(self.user_movie[u2][i])
            avg1 = user1/np.sum(user1)
            avg2 = user2/np.sum(user2)
            sum1 = np.sqrt(np.sum((user1-avg1)**2))
            sum2 = np.sqrt(np.sum((user2-avg2)**2))
            siml = np.sum((user1 - avg1)*(user2-avg2))/(sum1*sum2)
            self.similarity[u1][u2] = siml
            self.similarity[u2][u1] = siml
        return siml
        
    def pred(self,uid,mid):
        maxsim  = -2.0
        maxuser = 'nouser'
        for user in self.user_movie:
            if uid!=user and mid in self.user_movie[user]:
                sim = self.cal_sim(uid,user)
                if sim>maxsim:
                    maxsim = sim
                    maxuser = user
        if maxuser!='nouser':
            return self.user_movie[maxuser][mid]
        else:                
            return self.ave
    def test(self,test_X):  
        output=[]  
        sums=0  
        test_X=np.array(test_X)  
        #print "the test data size is ",test_X.shape  
        for i in range(100):  
            pre=self.pred(test_X[i][0],test_X[i][1])  
            output.append(pre)  
            #print pre,test_X[i][2]  
            sums+=(pre-float(test_X[i][2]))**2  
            print i
        rmse=np.sqrt(sums/100)  
        print "the rmse on test data is ",rmse  
        return output          

class  SVD_C:  
    def __init__(self,X,Y,k=30):  
        ''''' 
            k  is the length of vector 
        '''  
        self.X=np.array(X) 
        self.Y=Y
        self.k=k  
        self.ave=np.mean(self.X[:,2].astype(float))  
        print "the input data size is ",self.X.shape  
        self.bi={}  
        self.bu={}  
        self.qi={}  
        self.pu={}  
        self.movie_user={}  
        self.user_movie={}  
        for i in range(self.X.shape[0]):  
            uid=self.X[i][0]  
            mid=self.X[i][1]  
            rat=float(self.X[i][2])  
            self.movie_user.setdefault(mid,{})  
            self.user_movie.setdefault(uid,{})  
            self.movie_user[mid][uid]=rat  
            self.user_movie[uid][mid]=rat  
            self.bi.setdefault(mid,0)  
            self.bu.setdefault(uid,0)  
            self.qi.setdefault(mid,random((self.k,1))/10*(np.sqrt(self.k)))  
            self.pu.setdefault(uid,random((self.k,1))/10*(np.sqrt(self.k)))  
    def pred(self,uid,mid):  
        self.bi.setdefault(mid,0)  
        self.bu.setdefault(uid,0)  
        self.qi.setdefault(mid,np.zeros((self.k,1)))  
        self.pu.setdefault(uid,np.zeros((self.k,1)))  
        if (self.qi[mid]==None):  
            self.qi[mid]=np.zeros((self.k,1))  
        if (self.pu[uid]==None):  
            self.pu[uid]=np.zeros((self.k,1))  
        ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])  
        if ans>5:  
            return 5  
        elif ans<1:  
            return 1  
        return ans  
    def train(self,steps=40,gamma=0.04,Lambda=0.15):  
        for step in range(steps):  
            print 'the ',step,'-th  step is running'  
            rmse_sum=0.0  
            kk=np.random.permutation(self.X.shape[0])  
            for j in range(self.X.shape[0]):  
                i=kk[j]  
                uid=self.X[i][0]  
                mid=self.X[i][1]  
                rat=float(self.X[i][2])  
                eui=rat-self.pred(uid,mid)  
                rmse_sum+=eui**2  
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])  
                temp=self.qi[mid]  
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid])  
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])  
            gamma=gamma*0.93  
            print "the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0])  
            self.test(self.Y)
            #self.test(test_data)  
    def test(self,test_X):  
        output=[]  
        sums=0  
        test_X=np.array(test_X)  
        #print "the test data size is ",test_X.shape  
        for i in range(test_X.shape[0]):  
            pre=self.pred(test_X[i][0],test_X[i][1])
            output.append(pre)  
            #print pre,test_X[i][2]  
            sums+=(pre-float(test_X[i][2]))**2
        rmse=np.sqrt(sums/test_X.shape[0])  
        print "the rmse on test data is ",rmse  
        return output          

class  SVD_plusplus:  
    def __init__(self,X,k=20):  
        ''''' 
            k  is the length of vector 
        '''  
        self.X=np.array(X)  
        self.k=k  
        self.ave=np.mean(self.X[:,2].astype(float))  
        print "the input data size is ",self.X.shape  
        self.bi={}  
        self.bu={}  
        self.qi={}  
        self.pu={}  
        self.yi={}
        self.Ru={}
        self.movie_user={}  
        self.user_movie={}  
        for i in range(self.X.shape[0]):  
            uid=self.X[i][0]  
            mid=self.X[i][1]  
            rat=float(self.X[i][2])  
            if self.Ru.get(uid)==None:
                self.Ru[uid]=[mid]
            elif self.Ru[uid].count(mid)==0:
                    items = self.Ru[uid]
                    items.append(mid)
                    self.Ru[uid] = items
            self.movie_user.setdefault(mid,{})  
            self.user_movie.setdefault(uid,{})  
            self.movie_user[mid][uid]=rat  
            self.user_movie[uid][mid]=rat  
            self.bi.setdefault(mid,0)  
            self.bu.setdefault(uid,0)  
            self.qi.setdefault(mid,random((self.k,1))/10*(np.sqrt(self.k))) 
            self.yi.setdefault(mid,random((self.k,1))/10*(np.sqrt(self.k))) 
            self.pu.setdefault(uid,random((self.k,1))/10*(np.sqrt(self.k)))  
    def pred(self,uid,mid):  
        self.bi.setdefault(mid,0)  
        self.bu.setdefault(uid,0)  
        self.qi.setdefault(mid,np.zeros((self.k,1)))  
        self.pu.setdefault(uid,np.zeros((self.k,1)))  
#        if (self.qi[mid]==None):  
#            self.qi[mid]=np.zeros((self.k,1))  
#        if (self.pu[uid]==None):  
#            self.pu[uid]=np.zeros((self.k,1))
        if (self.Ru.get(uid)==None):    
            ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])
        else:
            yu = np.zeros((self.k,1))
            for vid in self.Ru[uid]:
                    yu+=self.yi[vid]
            yu = yu/np.sqrt(len(self.Ru[uid]))
            ans=self.ave+self.bi[mid]+self.bu[uid]+np.sum(self.qi[mid]*self.pu[uid])+np.sum(self.qi[mid]*yu)
        if ans>5:  
            return 5  
        elif ans<1:  
            return 1  
        return ans  
    def train(self,steps=10,gamma=0.04,Lambda=0.15):  
        for step in range(steps):  
            print 'the ',step,'-th  step is running'  
            rmse_sum=0.0  
            kk=np.random.permutation(self.X.shape[0])  
            for j in range(self.X.shape[0]):  
                i=kk[j]  
                uid=self.X[i][0]  
                mid=self.X[i][1]  
                rat=float(self.X[i][2])  
                eui=rat-self.pred(uid,mid)  
                rmse_sum+=eui**2  
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[mid]+=gamma*(eui-Lambda*self.bi[mid])  
                temp=self.qi[mid]  
                yu = np.zeros((self.k,1))
                for vid in self.Ru[uid]:
                    yu+=self.yi[vid]
                yu = yu/np.sqrt(len(self.Ru[uid]))
                self.qi[mid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[mid]+eui*yu)  
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])  
                self.yi[mid]+=gamma*(eui/np.sqrt(len(self.Ru[uid]))*self.qi[mid]-Lambda*self.qi[mid])
            gamma=gamma*0.93  
            print "the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0])  
            #self.test(test_data)  
    def test(self,test_X):  
        output=[]  
        sums=0  
        test_X=np.array(test_X)  
        #print "the test data size is ",test_X.shape  
        for i in range(test_X.shape[0]):  
            pre=self.pred(test_X[i][0],test_X[i][1])  
            output.append(pre)  
            #print pre,test_X[i][2]  
            sums+=(pre-float(test_X[i][2]))**2  
        rmse=np.sqrt(sums/test_X.shape[0])  
        print "the rmse on test data is ",rmse  
        return output          

a = SVD_plusplus(train_data)
a.train()  
a.test(test_data)