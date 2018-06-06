#!/usr/bin/env python
import sys
#sys.path.insert(0,'../POC/agent/')

#sys.path.insert(0,'/home/ubuntu/Desktop')

from NNAgentModified import *

import gym
import json

class Interpreter(object):

    def isValid(self, b):
        if b=='[' or b.isdigit():
            return False
        return True

    def isHigher(self,b):
        if b=='_':
            return False
        return True

    def collectorJson(self,read):
        #ful=address+'.json'
        #leer = json.loads(open(address).read())
        array = []
        result = []
        start = " "

        for key, value in read.items():

            converter = str(key)
            finds = converter.find("method")

            for clave, valor in value.items():
                if key == "class":
                    start = valor+"()"
                    operator = "a."

                if finds != -1:
                    param=""

                    if clave == "name":
                        method = valor

                    if clave=="parameters":
                        count = 0
                        order = []

                        for clav, val in valor.items():
                            order.append("")

                        for clav, val in valor.items():
                            clav=str(clav)
                            val = str(val)

                            if self.isValid(val[0]):
                                val="\""+val+"\""
                            if self.isHigher(clav[1]):
                                integ = clav[0:2]
                                integer = (int(integ))-1
                                order[integer]=str(val)
                            else:
                                id = (int (clav[0]))-1
                                order[id]=str(val)

                        for i in order:
                            param+=(i+",")

                        temp = len(param)
                        datos = param[:temp -1]
                        parameters = "("+datos+")"
                        funtion=method+parameters
                        array.append(funtion)

                    elif str(valor) == "getObservations":
                        funtion = method + "()"
                        array.append(funtion)

        if start != " ":
            a = eval(start)
            i = len(array)-1

            while i >=0:
                funtion = operator+array[i]
                eval (funtion)
                i-=1
        else:
            i = len(array)-1
            while i>=0:
                eval(array[i])
                i-=1
    
    def collectorString(self,stringS):
        print stringS
        stringS.pop(0)
        string =""
        #leer = json.dumps(stringS)
        #leer1 = json.loads(leer)
        #print leer1
        for i in stringS:
            string += i

        eval(string)


    def collectorPath(self,para):
        method = data[0]
        parameters = ""
        for i in data[1:len(data)]:
            parameters += i+","

        temp = len(parameters)
        datos = parameters[:temp -1]
        funtion = method+"("+datos+")"
        eval(funtion)

    def isFormat(self,data):
        #print data[0]
        if data[0]=='/' or data[0]=='.':
            #print "path"
            return 3
        if data[0]=='{' or data[0]=='\'':
            #print "json"
            return 2
        if data[0]=='s':
            #print "funtion"
            return 1

    def content(self, address):
        if len(address)==0:
            read = json.loads(open('./configuration.json').read())
            self.collectorJson(read)

        else:
            if self.isFormat(address[0])==3:
                read = json.loads(open(address[0]).read())
                self.collectorJson(read)

            elif self.isFormat(address[0])==2:
                read = json.loads(address[0])
                self.collectorJson(read)

            elif self.isFormat(address[0])==1:
                self.collectorPath(address)



if __name__== '__main__':
    a = Interpreter() 
    sys.argv.pop(0)
    a.content(sys.argv)

