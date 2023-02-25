# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:30:20 2023

@author: Yash Pungaliya
"""

#Parent or base class
class Bank:
    def __init__(self):
        self.bankname='Axis Bank'
        self.loc = 'MG Road'
        self.city = 'Pune'
        self.ifsc = 'AXIS0011111'
        
    def printBankInfo(self):
        print("bank={} , branch={},location ={}".format(self.bankname,self.loc,self.city))
    
#Create a child Class or derivd Class
class cCustomer(Bank):
    def __init__(self,cid,cname):
        super().__init__()
        self.cid = cid
        self.cname =cname
        self.acctype = 'Savings'
        self.balance = 1000.10
        self.l1 = []
        
    def printBank(self):
        super().printBankInfo()
        
    def depositMoney(self,n):
        self.printBalance()
        self.balance += n 
        self.printBalance()
        #l1.append
        
    def withdraw(self,n):
        if(n<self.balance):
            # print("After withdraw = {}".format(self.balance-n))
            print("in withdraw()")
            self.balance -= n
            self.printBalance()
        else:
            print("Not Sufficient balance")
            return
        
    def printBalance(self):
        print("account balance= {}".format(self.balance))
        
        
cust1 = cCustomer(123456, "Yash")
cust1.depositMoney(2000)
cust1.printBalance()

cust1.withdraw(10000)
