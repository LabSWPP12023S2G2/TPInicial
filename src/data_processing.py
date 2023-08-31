import pandas as pd
import numpy as np

def load_and_clean_data(url):
    # Carga y limpieza del dataset
    data = pd.read_csv(url, delimiter=';')
    columns_to_drop = ['SUB PERIODS', 'SEX']
    data = data.drop(columns=columns_to_drop)
    data = data.dropna(axis=0)
    data.drop(data[data['PROVINCE'] == 'Otro'].index, inplace=True)
    data.drop(data[data['PROVINCE'] == 'other'].index, inplace=True)
    data.drop(data[data['EDUCATION'] == 'Otro'].index, inplace=True)

    # Asignaciones para columnas no númericas
    assignment_mapping = {
        'MENTAL DISORDER HISTORY': {'no': 0, 'yes': 50},
        'EDUCATION': {
            'Completed postgraduate': 30,
            'Incomplete tertiary or university': 60,
            'Completed high school': 70,
            'Incomplete postgraduate': 40,
            'Completed tertiary or university': 50,
            'Incomplete high school': 80,
            'Incomplete elementary school': 100,
            'Completed elementary school': 90
        },
        'SUIC ATTEMPT HISTORY': {'ideation': 50, 'no': 0, 'yes': 100},
        'LIVING WITH SOMEBODY': {'no': 20, 'yes': 0},
        'ECONOMIC INCOME': {'yes': 0, 'no': 50}
    }

    # Aplicamos las asignaciones
    for column, mapping in assignment_mapping.items():
        data[column] = data[column].map(mapping)


    # Función para asignar una región a cada provincia
    def assign_region(province):
        if province in ['Corrientes', 'Chaco', 'Misiones', 'Formosa', 'Entre Ríos']:
            return 'Nordeste-Litoral'
        elif province in ['Tucumán', 'Jujuy', 'Salta', 'Catamarca', 'Santiago del Estero']:
            return 'Noroeste'
        elif province in ['San Luis', 'San Juan', 'Mendoza', 'La Rioja']:
            return 'Cuyo'
        elif province in ['Neuquén', 'Río Negro', 'La Pampa']:
            return 'Patagonia Centro-Norte'
        elif province in ['Tierra del Fuego', 'Santa Cruz', 'Chubut']:
            return 'Patagonia Centro-Sur'
        elif province == 'Santa Fe':
            return 'Santa Fe'
        elif province == 'Buenos Aires provincia':
            return 'Buenos Aires'
        elif province == 'Córdoba':
            return 'Córdoba'
        else:
            return 'CABA'

    # Aplicamos la función a la columna 'PROVINCE' y guardamos el resultado en una nueva columna 'REGION'
    data['REGION'] = data['PROVINCE'].apply(assign_region)


    # Descartamos las columnas que no usaremos momentaneamente
    columns_to_drop = ['REGION', 'PROVINCE', 'SUIC RISK']
    data = data.drop(columns=columns_to_drop)

    return data