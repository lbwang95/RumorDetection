#encoding='utf-8'
with open("weibo.txt","r",encoding='utf-8') as f:
    a=0
    for line in f:
        if a<=330*7:
            w1=open("weibo17.train","a",encoding='utf-8')
            w1.write(line)
        elif a<=330*9:
            w2=open("weibo17.dev","a",encoding='utf-8')
            w2.write(line)
        else:
            w3=open("weibo17.test","a",encoding='utf-8')
            w3.write(line)
        a+=1
w1.close()
w2.close()
w3.close()
          
          
