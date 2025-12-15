import numpy as np
import matplotlib.pyplot as plt
import csv

#Reading through the .csv file
with open("Admission_Predict_Ver1.1.csv", "r") as file:
    lines = file.readlines() 

#Inputting the values into a list
Serial = []
GRE = []
TOEFL = []
UniRat = []
SOP = []
LOR = []
CGPA = []
Research = []
ChanceAdmit = []

#Appending the values to the input parameter list
for line in lines[1:]:
    row = line.strip().split(",")
    Serial.append(float(row[0]))
    GRE.append(float(row[1]))
    TOEFL.append(float(row[2]))
    UniRat.append(float(row[3]))
    SOP.append(float(row[4]))
    LOR.append(float(row[5]))
    CGPA.append(float(row[6]))
    Research.append(float(row[7]))
    ChanceAdmit.append(float(row[8]))
    
#Creating a numpy array  
Serial = np.array(Serial)    
GRE = np.array(GRE)
TOEFL = np.array(TOEFL)
UniRat = np.array(UniRat)
SOP = np.array(SOP)
LOR = np.array(LOR)
CGPA = np.array(CGPA)
Research = np.array(Research)
ChanceAdmit = np.array(ChanceAdmit)


# Using the M matrix to perform linear regression by using the least squares of the defined function and minimising it
M = np.column_stack([GRE**1, TOEFL**1, CGPA**0, CGPA**2, CGPA**1, LOR**1, SOP**2, Research*LOR, TOEFL*UniRat**1])      
# Use the lstsq function to solve for the parameters
p, _, _, _ = np.linalg.lstsq(M, ChanceAdmit, rcond = None)    
print(f" The obtained equation is {p[0]}*GRE**1 + {p[1]}*TOEFL**1 + {p[2]}*CGPA**0 + {p[3]}*CGPA**2 + {p[4]}*CGPA**1 + {p[5]}*SOP**1 + {p[6]}*Research*LOR + {p[7]}*TOEFL*UniRat**1 ")

#Plotting the graph of the predicted chances of admit by the model versus the actual one
plt.scatter((p[0]*(GRE**1) + p[1]*(TOEFL**1) + p[2]*(CGPA**0) + p[3]*(CGPA**2) + p[4]*(CGPA**1) + p[5]*(LOR**1) +  + p[6]*(SOP**2) + p[7]*(Research*LOR) + p[8]*(TOEFL*UniRat**1)), ChanceAdmit )
plt.plot(ChanceAdmit, ChanceAdmit)

ActualAdmit = ChanceAdmit
PredicVal = p[0]*(GRE**1) + p[1]*(TOEFL**1) + p[2]*(CGPA**0) + p[3]*(CGPA**2) + p[4]*(CGPA**1) + p[5]*(SOP**2) + p[6]*(Research*LOR) + p[7]*(TOEFL*UniRat**1)
plt.plot(ActualAdmit, ChanceAdmit)
plt.savefig("image.png")
#Evaluating the value 
#CorrMatrix = np.corrcoef(ActualAdmit, ChanceAdmit)
#CorrCoeff = CorrMatrix[0, 1]
