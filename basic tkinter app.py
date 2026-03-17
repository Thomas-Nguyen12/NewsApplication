# tkinter app


import tkinter as tk
import joblib 
from tkinter import *

# This is the model script
from ai_detection import ai_detector




root = tk.Tk() 
root.geometry("1000x1000")



def detect():
    INPUT = inputtxt.get("1.0", 'end-1c')
    
    prediction = ai_detector(INPUT).predict() 

    prediction_confidence = ai_detector(INPUT).predict_proba() 
    Output.delete("1.0", END)
    Output.insert(END, prediction)
    Output.insert(END, "\n")
    Output.insert(END, "Confidence:")
    Output.insert(END, prediction_confidence) 

label = tk.Label(root, text="AI text detector")
label.pack()
label2 = tk.Label(root, text="Please enter your text below: ")
label2.pack() 


inputtxt = tk.Text(root, height=30, width=60) 

Output = tk.Text(root, height=10, width=30)
Display = tk.Button(root, text='Enter', command=lambda:detect()) 

inputtxt.pack()
Display.pack()
Output.pack()



root.mainloop() 
