import os
os.system("cd train; rm *.*; cd ..")
os.system("cd val; rm *.*; cd ..")
os.system("cd train_h5; rm *.*; cd ..")
os.system("cd val_h5; rm *.*; cd ..")
os.system("rm train.list; rm val.list; rm train_hdf5.txt; rm validation_hdf5.txt")

