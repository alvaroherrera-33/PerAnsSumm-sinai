import json
import base64
import html
# Cargar el archivo JSON
with open("/mnt/beegfs/aarjonil/peranssumm/PerAnsSumm-sinai/dataset/train.json", "r", encoding="utf-8") as file:
    data = json.load(file)
kk="uri: 1857842\nquestion: What is the best way to fight off the flu without a shot?\ncontext: \nanswer_0: Wash your hands before you eat anything, or better yet use an antibacterial that you can keep in your purse, or if you're a guy in your pocket, don't put your fingers in your mouth (i.e., bite your nails). One good piece of advice in preventing colds or the flu is to not borrow pens...always make sure you have one of your own...that way you know that you're not holding something that someone who just coughed or sneezed into their hand was just touching.As always, make sure you take in lots of vitamin C daily.  Lots of orange juice, and go with a multivatimin.  FYI, all of these things are good to practice anyway but are especially good practices to remember around flu season.\nanswer_1: Eat Healthy. soups, fruit... Stay in bed, dont leave your house...\nSUGGESTION_GROUP: ash your hands before you eat anything, or better yet use an antibacterial that you can keep in your purse, or if you're a guy in your pocket, don't put your fingers in your mouth (i.e., bite your nails). One good piece of advice in preventing colds or the flu is to not borrow pens...always make sure you have one of your own...that way you know that you're not holding something that someone who just coughed or sneezed into their hand was just touching.As always, make sure you take in lots of vitamin C daily.  Lots of orange juice, and go with a multivatimin..Eat Healthy. soups, fruit... Stay in bed, dont leave your house\n"
# Usamos json.dumps con ensure_ascii=True para forzar la codificación Unicode
#texto_codificado = json.dumps(data[0], ensure_ascii=True)
print(data[11])
print()
#kk=kk.b64encode("unicode_escape")
kk= html.escape(kk)
print(kk)
print()
print(kk[102:668])