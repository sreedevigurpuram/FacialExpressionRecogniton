import tkinter as tk
from PIL import ImageTk, Image
from camera import *
import sqlite3,csv
from tkinter import messagebox

window = tk.Tk()
window.title("Face Emotion")
window.configure(background='Black')


def Detect():
   
        result=main()
        con = sqlite3.connect('facial_emotion.db')
        cur = con.cursor()

        print("result",result)
        print("type",type(result))
        
        x=list(result.keys())
        print(x)
        x.sort()
        sql='''INSERT INTO emotions(angry,disgust,fear,happy,sad,surprise,neutral)
            values('%s','%s','%s','%s','%s','%s','%s')''' %(result["Angry"],result["Disgust"],result["Fear"],result["Happy"],result["Sad"],result["Surprise"],result["Neutral"],)
        cur.execute(sql)
        con.commit()
        con.close()
        import matplotlib.pyplot as plt; plt.rcdefaults()
        import numpy as np
         
        objects = tuple(result.keys())
        y_pos = np.arange(len(objects))
        performance = result.values()
         
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.xlabel('Emotions')
        plt.ylabel('Frames')
        plt.title('Emotion Analysis')
        plt.show()

path = "index.jpg"

img = ImageTk.PhotoImage(Image.open(path))

panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")




b2=tk.Button(panel,text="Start Analysis", bg="green",width=20,height=2,command=Detect)
b2.pack(side="left")
b2.place(x=300,y=200)



window.mainloop()
