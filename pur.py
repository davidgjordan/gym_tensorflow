
import sys
import json

path_constant = sys.argv[1] #./Games/pacman/t1/game.json
pathAux = path_constant

arrayPath = pathAux.split('/')
res = ""
for i in range(len(arrayPath)-1):
    res+=arrayPath[i] 
    if(i != len(arrayPath)-2 ):
        res+="/"
print res