import uvicorn
import re
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import sklearn
import numpy as np

app = FastAPI()


class Item(BaseModel):
    name: str | None
    year: int | None
    selling_price: int | None
    km_driven: int | None
    fuel: str | None
    seller_type: str | None
    transmission: str | None
    owner: str | None
    mileage: str | None
    engine: str | None
    max_power: str | None
    torque: str | None
    seats: float | None


class Items(BaseModel):
    objects: List[Item]
    
    
class Model:
    
    def load_model(self):
        with open("fill_df.pickle", "rb") as f:
            self.fill_df = pickle.load(f)
        with open("encoder.pickle", "rb") as f:
            self.enc = pickle.load(f)
        with open("scaler.pickle", "rb") as f:
            self.scal = pickle.load(f)
        with open("ridge.pickle", "rb") as f:
            self.ridge = pickle.load(f)
            
    def predict(self, items: Items) -> List[float]:
        df = pd.DataFrame([dict(item) for item in items.objects])
        self.fill_predictions_in_dataframe(df)
        result = df['selling_price'].to_list()
        return result
    
    def clean_torque(self, df):
        
        df = df.copy()
    
        df['torque'] = df['torque'].astype(str).str.replace(r'[, ]', '', regex=True)
    
        patterns = [r'(.*)@', r'(.*)at(.*)', r'(.*)kgm(.*)','(.*)@ ']
    
        df['Torque'] = ''
        df['max_torque_rpm'] = ''
    
        # Iterate through the patterns to extract parts
        for pattern in patterns:
            part1 = df['torque'].str.extract(pattern)[0]
        
            # Update Part1 and Part2 if a match is found
            df.loc[part1.notnull(),'Torque'] = part1
    
        # Remove the units from the extracted values
        df['Torque']= df['Torque'].str.replace('Nm', '')
        df['Torque']= df['Torque'].str.replace('kgm', '')
        df['Torque']= df['Torque'].str.replace('KGM', '')
        df['Torque']= df['Torque'].str.replace('nm', '')
    
        # Iterate through the patterns to extract parts
        for pattern in patterns:
            part1 = df['Torque'].str.extract(pattern)[0]
            df.loc[part1.notnull(),'Torque'] = part1
        
        # Define a function to extract the 4-digit number before "rpm" or "(kgm@rpm)" in a string
        def extract_torque(text):
            match = re.search(r"(\d{4})(rpm|(kgm@rpm))", text)
            if match:
                return int(match.group(1))
            return None
    
        # Apply the function to the 'torque' column and create a new 'max_torque_rpm' column
        df['max_torque_rpm'] = df['torque'].apply(extract_torque)
    
        # Convert the extracted values to their respective data types
        df['Torque'] = pd.to_numeric(df['Torque'], errors='coerce')
    
        # Drop the original 'torque' column and rename the new one
        df.drop('torque', axis=1, inplace=True)
        df.rename(columns={"Torque": "torque"}, inplace=True)
    
        return df
    
    def fill_predictions_in_dataframe(self, get_df: pd.DataFrame) -> None:
        # your pipeline code here
        df = get_df.copy()
        if 'selling_price' in df.columns:
            df.drop('selling_price', axis =1, inplace=True)
        #РџСЂРёРІРµРґРµРЅРёРµ Рє С‡РёСЃР»РѕРІС‹Рј Р·РЅР°С‡РµРЅРёСЏРј
        df['mileage'] = pd.to_numeric(df['mileage'].replace('kmpl|km/kg', '', regex=True).str.strip()).astype(float)
        df['engine'] = pd.to_numeric(df['engine'].replace('CC', '', regex=True).str.strip()).astype(float)
        df['max_power'] = pd.to_numeric(df['max_power'].replace('bhp', '', regex=True).str.strip()).astype(float)
        
        df = self.clean_torque(df)

        #Р—Р°РїРѕР»РЅРµРЅРёРµ РјРµРґРёР°РЅРЅС‹РјРё
        cols = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
        df[cols] = self.fill_df.transform(df[cols])
        
        df = df.astype({'engine': 'int', 'seats': 'int'})

        #РќРѕРІС‹Рµ РїРѕР»СЏ
        df['age'] = 2021 - df['year']
        df.drop(['year'],axis = 1,inplace = True)
        df['owner'] = df['owner'].replace({'Test Drive Car': 0, 'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner':4})

        df['brand'] = df['name'].str.split(' ').str.get(0)
        df.drop(['name'], axis=1, inplace=True)
        # df['selling_price'] = np.log(df['selling_price'])
        df['max_power'] = np.log(df['max_power'])
        df['age'] = np.log(df['age'])

        #OneHotEncoding
        num_var = ['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'age']
        cat_cols = ['seats', 'fuel', 'seller_type', 'transmission', 'owner', 'brand']
        df[num_var] = self.scal.transform(df[num_var])

        df_enc = pd.DataFrame(self.enc.transform(df[cat_cols]).toarray(),
                                    columns=self.enc.get_feature_names_out(cat_cols), dtype=int)
        X= pd.concat([df.drop(columns=cat_cols), df_enc], axis=1)

        #РџСЂРµРґСЃРєР°Р·Р°РЅРёРµ
        y_pred = self.ridge.predict(X)
        y_pred = np.exp(y_pred)
        get_df['selling_price'] = y_pred        
    
model = Model()
model.load_model()

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict(Items(objects=[item]))[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return model.predict(Items(objects=items))

@app.post('/predict_items_from_csv_file')
def upload_csv(file: UploadFile) -> FileResponse:
    df = pd.read_csv(file.file)
    model.fill_predictions_in_dataframe(df)

    df.to_csv('results.csv')
    response = FileResponse(path='results.csv', media_type='text/csv', filename='results.csv')
    return response