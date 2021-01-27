# Mexute Savarjisho
import numpy as np
from numpy import inf
from numpy.linalg import norm
import statistics as st
from scipy.stats import skew
from sklearn.linear_model import ElasticNet

a = np.array([4,-1,6])
b = np.array([3,9,5])
c = np.array([5,7,-4])

# ა) იპოვეთ ჯამი და სხვაობა
add_vectors = a + b
sub_vectors = a - b
print("Jami = ", add_vectors)
print("Sxvaoba = ", sub_vectors)

# ბ) იპოვეთ ვექტორების სკალარული ნამრავლი
Skalaruli_Namravli = np.dot(a,b)
print("Skalaruli Namravli = ", Skalaruli_Namravli)

# გ) იპოვეთ ვექტორების ვექტორული ნამრავლი
cross_product = np.cross(a,b)
print("Vektoruli Namravli = ", cross_product)

# დ) მოცემულია მესამე ვექტორი c = (5,7,-4). იპოვეთ ვექტორების შერეული ნამრავლი
mixed_product = np.dot(np.cross(a,b),c)
print("Shereuli Namravli = ", mixed_product)

# ე) იპოვეთ a ვექტორის სიგრძე
norm_vector_a = norm(a)
print("A vectoris Sigrdze = ", norm_vector_a)

# ვ) იპოვეთ ნულოვანი რიგის ნორმა b ვექტორისთვის
norm_vector_b_nulovani_rigi = norm(b,0)
print("B vectoris Nulovani rigis norma = ", norm_vector_b_nulovani_rigi)

# ზ) იპოვეთ პირველი რიგის ნორმა b ვექტორისთვის
norm_vector_b_pirveli_rigi = norm(b,1)
print("B vectoris Pirveli Rigis norma = ", norm_vector_b_pirveli_rigi)

# თ) იპოვეთ მეორე რიგის ნორმა b ვექტორისთვის
norm_vector_b_meore_rigi = norm(b,2)
print("B vectoris Meore Rigis norma = ", norm_vector_b_meore_rigi)

# ი) იპოვეთ უსასრულო რიგის ნორმა b ვექტორისთვის
norm_vector_b_usasrulo_rigi = norm(b, inf)
print("B Vectoris usasrulo rigis norma = ", norm_vector_b_usasrulo_rigi)

# კ) იპოვეთ ვექტორს შორის მანძილი
distance_between_vectors = norm(b-a)
print("Vectors shoris mandzili = ", distance_between_vectors)


#Meeqvse Savarjisho
A = np.array([[2,5,4],[-3,6,-2],[1,4,5]])
B = np.array([34,-4,35])

solution_A_B = np.linalg.solve(A,B)
print("Amoxsna = ", solution_A_B)

#Meshvide Savarjisho
huge_array = np.array([25,27,21,13,29,13,8,12,23,18,7,7,29,30,30,8,24,29,2,19,11,5,28,3,5,22,4,23,29,3,24,30,29,6,1])

print(np.amin(huge_array)) #Minimum
print(np.amax(huge_array)) #Maximum
print(np.ptp(huge_array)) #Gani
print(np.median(huge_array)) #Mediana
print(np.mean(huge_array)) #Sashualo aritmetikuli
print(np.std(huge_array)) #Standartuli gadaxra
print(np.var(huge_array)) #Variacia/Dispersia
print(st.mode(huge_array)) #Moda
print(skew(huge_array)) #Asimetriuloba

#Merve Savarjisho
x = np.array([[118,105],[28,56],[17,54],[50,63],[56,28],[102,50],[116,54],[124,42]])
y = np.array([203,63,45,113,121,88,110,56])
model = ElasticNet(alpha=1.0,l1_ratio=0.5)
model.fit(x,y)
r_sq = model.score(x,y)

print('coefficient of determination', r_sq)

print('intercept: ', model.intercept_)
print('Slope: ', model.coef_)
y_pred = model.predict(x)
print('Predicted Y is: ', y_pred)