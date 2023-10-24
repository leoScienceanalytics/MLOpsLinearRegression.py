from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from   sklearn.linear_model import LinearRegression

x = []
y = []


regressor = LinearRegression()

seed = 1234567
kf = KFold(n_splits=4, shuffle=True, random_state=seed)


i = 0
for train, test in kf.split():
    i = i + 1
    print(f'Separação: {i}: \Treino: {[]} \n\n Teste: {[]}\n')
    
    
scores = cross_val_score(regressor, x, y, cv=kf, scoring='neg_mean_absolute_error')
mae_scores=-scores
mean_mae= mae_scores.mean()
