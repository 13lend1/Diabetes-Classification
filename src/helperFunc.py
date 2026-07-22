import os
import pandas as pd
def saveFile(file,path):
    file.to_csv(path,index=False)
    
    