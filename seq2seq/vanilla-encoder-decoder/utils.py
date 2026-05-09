def create_data(file_path):

  with open(file_path,"r") as f:
    file_text = f.read()

  file_data = file_text.split("\n")

  X = []
  y = [] 

  for i in file_data:
    x_y = i.split("\t")
    if(len(x_y)>1):
      X.append(x_y[0])
      y.append(x_y[1])


  return X,y