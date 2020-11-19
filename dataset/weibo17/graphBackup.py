import random
dic={}
with open("weibo17_graph_ori.txt", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tmp = line.strip().split()
        dic[tmp[0]]=10000000
        for j in tmp[1:]:
            dst,w = j.split(":")
            if dst in dic:
                dic[dst]+=1
            else:
                dic[dst]=1
w=open("weibo17_graph.txt","w", encoding='utf-8')
with open("weibo17_graph_ori.txt", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tmp = line.strip().split()
        if len(tmp)>2:
            num=100000000#int(0.2*len(tmp))
            j=0
            w.write(tmp[0]+" ")
            for i in range(1,len(tmp)):
                if dic[tmp[i].split(':')[0]]>3:
                    w.write(tmp[i]+" ")
                    j+=1
                    if j==num:
                        break
            w.write("\n")
        else:
            w.write(line)
w.close()
                
