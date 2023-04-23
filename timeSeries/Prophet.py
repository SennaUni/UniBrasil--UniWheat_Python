import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import dateutil.relativedelta as rdt
import sys
from prophet import Prophet

# Recebe como parametro a quantidade de meses
# months = sys.argv[1]
months = 8

# Configurações de display de dataFrames
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Importando e formatando dados da tabela de trigo
dataWheat = pd.read_csv('timeSeries/data/wheatData.csv', parse_dates = True)

# Apagando colunas com dados que não serão utilizados
dataWheat.drop('Daily %', axis = 1, inplace = True)
dataWheat.drop('Monthly %', axis = 1, inplace = True)

# Definindo o nome para as colunas restantes
dataWheat.columns = [
    'ds', 
    'y'
]

# Visualizar variação de valores ao passar dos anos
# dataWheat.plot()
# plt.show()

# Formatando datas do formato m/d/Y para Y-m-d
dataWheat['ds'] = dataWheat['ds'].map(lambda x: dt.datetime.strptime(x, "%m/%d/%Y").strftime("%Y-%m-%d"))

# Importando e formatando dados da tabela de clima
dataWeather = pd.read_csv('timeSeries/data/weatherData.csv', parse_dates = True)

# Definindo o nome para as colunas
dataWeather.columns = [
    'ds', 
    'Code', 
    'Max', 
    'Min', 
    'Mean', 
    'MaxApparent', 
    'MinApparent', 
    'MeanApparent',
    'Sunsire',
    'Sunset',
    'Radiation',
    'Precipitation',
    'Rain',
    'SnowFall',
    'PrecipitationHour',
    'MaxWindSpeed',
    'MaxWindGusts',
    'WindDirection',
    'Evapotranspiration',
]

# Apagando colunas com dados que não serão utilizados
dataWeather.drop('Code', axis = 1, inplace = True)

# Importando e formatando dados da tabela de queimadas
dataBurned = pd.read_csv('timeSeries/data/BurnedData.csv', parse_dates = True)

# Apagando colunas com dados que não serão utilizados
dataBurned.drop('satelite', axis = 1, inplace = True)
dataBurned.drop('pais', axis = 1, inplace = True)
dataBurned.drop('estado', axis = 1, inplace = True)
dataBurned.drop('municipio', axis = 1, inplace = True)
dataBurned.drop('precipitacao', axis = 1, inplace = True)

# Definindo o nome para as colunas
dataBurned.columns = [
    'ds',
    'Biome',
    'WithoutRain',
    'FireRisc',
    'Latitude',
    'Longitude',
    'FirePower',
]

# Formatando datas do formato m/d/Y para Y-m-d
dataBurned['ds'] = dataBurned['ds'].map(lambda x: dt.datetime.strptime(x, "%m/%d/%Y").strftime("%Y-%m-%d"))

# dataBurned['Latitude'] = dataBurned['Latitude'].map(lambda x: x.replace('.', '').replace(',', '.'))
# dataBurned['Longitude'] = dataBurned['Longitude'].map(lambda x: x.replace('.', '').replace(',', '.'))

dataBurned = dataBurned.dropna().copy()

# Agrupando por Date os valores das diversas bases CSV para um unico dataFrame
allData = pd.merge(dataWheat, dataWeather, on = 'ds', how = 'inner')
allData = pd.merge(allData, dataBurned, on = 'ds', how = 'left')

# Convertendo o campo ds para DateTime
allData['ds'] = pd.to_datetime(allData['ds']) 

#print(allData)

# Definindo dados para treino e teste
# dataSize = int(len(allData) * 0.8)
# dataTrain = allData[:dataSize]
# dataTest  = allData[dataSize:]

#print(allData)
#print(dataBurned)

actualDate = pd.to_datetime('2020-01-01', format = '%Y-%m-%d')

calcDate = actualDate + rdt.relativedelta(months = int(months))

dataTrain = allData.loc[
    # (allData['ds'] >= pd.to_datetime('2010-01-01', format = '%Y-%m-%d'))
    # &
    (allData['ds'] < pd.to_datetime('2020-01-01', format = '%Y-%m-%d'))
].copy()

dataTest = allData.loc[
    (allData['ds'] >= pd.to_datetime('2020-01-01', format = '%Y-%m-%d'))
    &
    (allData['ds'] <= calcDate)
].copy()


# Criando o modelo e configurando dados de TREINO
model = Prophet()

model.add_regressor('Max')
model.add_regressor('Min')
model.add_regressor('Mean')
model.add_regressor('MaxApparent')
model.add_regressor('MinApparent')
model.add_regressor('MeanApparent')
# model.add_regressor('Sunsire')
# model.add_regressor('Sunset')
model.add_regressor('Radiation')
model.add_regressor('Precipitation')
model.add_regressor('Rain')
model.add_regressor('SnowFall')
model.add_regressor('PrecipitationHour')
model.add_regressor('MaxWindSpeed')
model.add_regressor('MaxWindGusts')
model.add_regressor('WindDirection')
model.add_regressor('Evapotranspiration')
# model.add_regressor('Biome')
# model.add_regressor('WithoutRain')
# model.add_regressor('FireRisc')
# model.add_regressor('Latitude')
# model.add_regressor('Longitude')
# model.add_regressor('FirePower')

# Treinando nosso modelo
model.fit(dataTrain)

# Cria um dataframe com as previsões para o período de teste
future = model.make_future_dataframe(periods = len(dataTest), freq = 'D')

future['Max'] = allData['Max']
future['Min'] = allData['Min']
future['Mean'] = allData['Mean']
future['MaxApparent'] = allData['MaxApparent']
future['MinApparent'] = allData['MinApparent']
future['MeanApparent'] = allData['MeanApparent']
future['Radiation'] = allData['Radiation']
future['Precipitation'] = allData['Precipitation']
future['Rain'] = allData['Rain']
future['SnowFall'] = allData['SnowFall']
future['PrecipitationHour'] = allData['PrecipitationHour']
future['MaxWindSpeed'] = allData['MaxWindSpeed']
future['MaxWindGusts'] = allData['MaxWindGusts']
future['WindDirection'] = allData['WindDirection']
future['Evapotranspiration'] = allData['Evapotranspiration']
# future['Biome'] = allData['Biome']
# future['WithoutRain'] = allData['WithoutRain']
# future['FireRisc'] = allData['FireRisc']
# future['Latitude'] = allData['Latitude']
# future['Longitude'] = allData['Longitude']
# future['FirePower'] = allData['FirePower']

# Realizando predições com a base de teste
forecast = model.predict(future)

# Plota o gráfico com os dados históricos e as previsões futuras
model.plot(forecast, xlabel = 'Data', ylabel = 'Preço')
plt.plot(dataTest['ds'], dataTest['y'], 'r', label='Dados Reais')
plt.legend()
plt.title(f"Estimativa do preço do trigo para {months} meses")
plt.show()