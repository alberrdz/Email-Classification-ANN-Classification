from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# lectura de datos
x = np.loadtxt('emails_spam.csv', delimiter=',', usecols=range(3000))
y = np.loadtxt('emails_spam.csv', delimiter=',', usecols=(3000, ))
print('Lectura de archivo de entrenamiento completada...')

# split 70% entrenamiento / 30% prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=5)

# entrenamiento
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=6, max_iter=1500)
clf.fit(x_train, y_train)
print('\nEntrenamiento completado...')

# predicción
predicted = clf.predict(x_test)

# cálculo del error 
error = 1 - accuracy_score(y_test, predicted)
print ('\nError en conjunto de prueba (entrenamiento):', error)

# guardado del modelo
filename = 'trained_model_EMAILSPAM.sav'
pickle.dump(clf, open(filename, 'wb'))

# lectura de datos para test y predicción
xS = np.loadtxt('emails_spamSubset.csv', delimiter=',', usecols=range(3000))
yS = np.loadtxt('emails_spamSubset.csv', delimiter=',', usecols=(3000, ))
print('\nLectura de archivo de prueba (predicción) completada...')

#carga del modelo
loaded_model = pickle.load(open('trained_model_EMAILSPAM.sav', 'rb'))

# predicción
newPredicted = loaded_model.predict(xS)

# impresión de la predicción de clasificación y el valor real de la clasificación
print('\nLas clasificaciones predichas por la RNA son: \n', yS)
print('\nLas clasificaciones reales del conjunto de datos son: \n', newPredicted)
print('\nDonde 1 -> spam, 0 -> libre de spam')

# cálculo del error de predicción
newError = 1 - accuracy_score(yS, newPredicted)
print ('\nError entre predicción y dato real del dataset:', newError)
