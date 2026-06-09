"""Эмпирическое восстановление шаблона keypoints модели DogFLW.

Усредняет нормализованные позиции каждого из 46 каналов по многим
фронтальным мордам (TTA выключен → сырой порядок каналов модели).
Сохраняет mean-template и рендер с номерами каналов.
"""
import json, subprocess, collections, re, os
import numpy as np, cv2
from packages.pipeline import InferencePipeline, PipelineConfig
from packages.models.keypoints import KeypointsConfig, KeypointsModel
from packages.data.schemas import NUM_KEYPOINTS

def curl(u): return subprocess.run(["curl","-s",u],capture_output=True,text=True,timeout=40).stdout
def curlb(u,fn): subprocess.run(["curl","-s","-o",fn,u],timeout=40)

pipe=InferencePipeline(PipelineConfig(device="cpu",keypoints_two_pass=True)); pipe.load()
# отдельная модель keypoints БЕЗ TTA (чтобы не мешать ложным flip-парам)
km=KeypointsModel(KeypointsConfig(weights_path="models/keypoints_dogflw.pt",device="cpu",use_tta=False)); km.load()

# породы с явной фронтальной мордой
breeds=["pug","beagle","boxer","labrador","bulldog/french","retriever/golden",
        "chihuahua","pomeranian","husky","rottweiler","spaniel/cocker","setter/english",
        "dalmatian","doberman","greatdane","bullterrier","pointer/german","weimaraner",
        "newfoundland","mastiff/bull"]
os.makedirs("data/tmpl",exist_ok=True)
acc=np.zeros((NUM_KEYPOINTS,2)); accv=np.zeros(NUM_KEYPOINTS); n_used=0
for b in breeds:
    for k in range(4):
        try:
            r=json.loads(curl(f"https://dog.ceo/api/breed/{b}/images/random"))
            if r.get("status")!="success": continue
            fn=f"data/tmpl/{b.replace('/','_')}_{k}.jpg"; curlb(r["message"],fn)
            img=cv2.imread(fn)
            if img is None: continue
            dets=pipe.bbox_model.predict(img)
            if not dets: continue
            x,y,w,h=dets[0].bbox
            # tight face crop через two-pass региона
            c1,ox1,oy1=pipe._square_crop(img,x,y,w,h,1.1)
            p1=km.predict(c1)
            fx,fy,fw,fh=pipe._keypoints_face_region(p1.keypoints)
            c2,ox2,oy2=pipe._square_crop(img,int(fx+ox1),int(fy+oy1),int(fw),int(fh),1.6)
            if c2.size==0 or min(c2.shape[:2])<=10: continue
            H,W=c2.shape[:2]
            p2=km.predict(c2)
            vis=np.array([kp.visibility for kp in p2.keypoints])
            if np.median(vis)<0.2: continue
            for i,kp in enumerate(p2.keypoints):
                wgt=max(kp.visibility,0.0)
                acc[i,0]+=wgt*(kp.x/W); acc[i,1]+=wgt*(kp.y/H); accv[i]+=wgt
            n_used+=1
        except Exception as e:
            pass
print(f"USED_FACES={n_used}")
mean=np.zeros((NUM_KEYPOINTS,2))
for i in range(NUM_KEYPOINTS):
    if accv[i]>0: mean[i]=acc[i]/accv[i]
json.dump(mean.tolist(), open("data/kp_template.json","w"))
# рендер шаблона
S=600; canvas=np.full((S,S,3),255,np.uint8)
for i in range(NUM_KEYPOINTS):
    px,py=int(mean[i,0]*S),int(mean[i,1]*S)
    cv2.circle(canvas,(px,py),4,(0,0,200),-1)
    cv2.putText(canvas,str(i),(px+3,py-3),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
cv2.imwrite("data/kp_template.png",canvas)
print("saved data/kp_template.png data/kp_template.json")
