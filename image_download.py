import requests
import os

muffin_images=[]
with open('muffins.txt', 'r') as file:
    for line in file:
        muffin_images.append(line)

count=0
dir='Images/chihuahua-muffin/train/muffins'
for url in muffin_images[:1000]:
    try:
        count+=1
        print(count)
        filename=os.path.join(dir,str(count)+'.jpg')
        f = open(filename, 'wb')
        f.write(requests.get(url).content)
        f.close()
    except:
        print('failer')