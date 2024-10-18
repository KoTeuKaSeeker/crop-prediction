import requests
import pandas as pd
import os

save_path = r"data\argo_dataset\argo_dataset.csv"
weather_parameters = ["SOLAR_RADIATION",
                        "PRECIPITATION",
                        "WIND_SPEED",
                        "LEAF_WETNESS",
                        "HC_AIR_TEMPERATURE",
                        "HC_RELATIVE_HUMIDITY",
                        "DEW_POINT"]

def load_agro_dataset(save_path: str):
    df = pd.DataFrame()
    for parameter_name in weather_parameters:
        response = requests.post('https://agroapi.xn--b1ahgiuw.xn--p1ai/parameter/', json={
            "endTime": 1729014252,
            "meteoId": "00001F76",
            "parameterName": parameter_name,
            "startTime": 0,
            "timestamp": True,
            "intervals": "day"
        })
        
        assert response.status_code == 200, f"Request error: status code {response.status_code}"

        data = response.json()
        if "date" not in df:
            df["date"] = data["dates"]
        df[parameter_name] = data["values"]["values"]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    load_agro_dataset(save_path)