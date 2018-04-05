'''
Mit diesem Script können Bilder direkt heruntergeladen werden.
Dazu muss eine txt Datei mit URL Pfadangaben zu Bildern in dem 
Projektpfad vorhanden sein. Aus diesen Pfaden werden Bilder herunter-
geladen und in einen Ordner gespeichert. 

'''
import requests
import os

filename = "Your_File.txt"

images=[]
with open(filename, 'r') as file:
    for line in file:
        images.append(line)

count = 0
className = 'Class_A'
dir='Images/train/'+className
for url in images[:1000]:
    try:
        count+=1
        print(count)
        image_file_name=os.path.join(dir,className+'_'+str(count)+'.jpg')
        f = open(image_file_name, 'wb')
        f.write(requests.get(url).content)
        f.close()
    except:
        print('failer')