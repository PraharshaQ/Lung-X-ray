#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:50:16 2018

@author: praharsha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:17:23 2018

@author: praharsha
"""

#import PIL
from PIL import Image
import os
import sys
def readf():
    try:
        input_dir  = str(sys.argv[1].rstrip('/'))
        #print(os.listdir( input_dir ))#path to img source folder
        img_size   = str(sys.argv[2])  #The image size (128, 256,etc)
        output_dir  = str(sys.argv[3].rstrip('/')) #output directory
        print "starting...."
        print "Colecting data from %s " % input_dir
        tclass = [ d for d in os.listdir( input_dir ) ]
        counter = 0
        for x in tclass:
           list_dir =  os.path.join(input_dir, x )
           print list_dir
           #list_tuj = os.path.join(output_dir+'/', x)
           #print(list_tuj)
           if not os.path.exists(output_dir):
                os.makedirs(output_dir)
           if os.path.exists(output_dir):
               try:
                   img = Image.open(list_dir)
                   
                   img = img.resize((int(img_size),int(img_size)),Image.ANTIALIAS)
                   
                   fname,extension = os.path.splitext(list_dir)
                   newfile = fname+extension
                   if extension != ".png" :
                       newfile = fname + ".png"
                   img.save(os.path.join(output_dir,x),"png",quality=90)
                   print "Resizing file : %s - %s " % (x,list_dir)
               except Exception as e:
                    print "Error resize file : %s - %s " % (x,list_dir)
                    sys.exit(1)
               counter +=1
    except Exception as e:
        print "Error, check Input directory etc : ", e
        sys.exit(1)
readf()
