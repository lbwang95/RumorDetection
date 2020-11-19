#encoding='utf-8'
import random
with open("weibo.txt","r",encoding='utf-8') as f:
    fi=[]
    for line in f:
        fi.append(line)
    random.shuffle(fi)
    a=0
    for line in fi:
        if a<=330*7:
            w1=open("weibo17.train","a",encoding='utf-8')
            w1.write(line)
        elif a<=330*8:
            w2=open("weibo17.dev","a",encoding='utf-8')
            w2.write(line)
        else:
            w3=open("weibo17.test","a",encoding='utf-8')
            w3.write(line)
        a+=1
w1.close()
w2.close()
w3.close()
          
          
