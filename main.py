from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pickle
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.json", "r", encoding="utf-8") as f:
    enc_data = json.load(f)

le_tip = LabelEncoder().fit(enc_data["arac_tipleri"])
le_marka = LabelEncoder().fit(enc_data["markalar"])
le_sehir = LabelEncoder().fit(enc_data["sehirler"])

app = FastAPI(
    title="Araç Risk Skorlama API",
    description="""
## Türkiye Araç Kasko/Trafik Risk Skorlama API

Makine öğrenmesi tabanlı araç risk skoru hesaplama.
Sigorta acenteleri, insurtech şirketleri ve araç kiralama firmaları için.

### Kullanım alanları
- Kasko prim hesaplama tabanı
- Trafik sigortası risk değerlendirme
- Araç kiralama risk filtresi
- Portföy analizi

### Skor yorumu
| Skor | Risk Seviyesi | Tavsiye |
|------|--------------|---------|
| 0–25 | Düşük | Standart prim |
| 26–50 | Orta | +%15–30 prim |
| 51–75 | Yüksek | +%30–60 prim |
| 76–100 | Çok Yüksek | Ret veya +%60+ prim |
    """,
    version="1.0.0",
    contact={"name": "API Desteği", "email": "destek@example.com"}
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RiskIstegi(BaseModel):
    arac_tipi: str = Field(..., description="Araç tipi", examples=["Sedan"])
    marka: str = Field(..., description="Araç markası", examples=["Toyota"])
    arac_yasi: int = Field(..., ge=0, le=30, description="Aracın yaşı (yıl)", examples=[5])
    motor_hacmi: float = Field(..., ge=0.8, le=6.0, description="Motor hacmi (litre)", examples=[1.6])
    surucu_yasi: int = Field(..., ge=18, le=90, description="Sürücü yaşı", examples=[35])
    surucu_deneyim: int = Field(..., ge=0, le=70, description="Sürüş deneyimi (yıl)", examples=[10])
    kaza_gecmisi: int = Field(..., ge=0, le=20, description="Son 5 yıldaki kaza sayısı", examples=[0])
    trafik_cezasi: int = Field(..., ge=0, le=50, description="Son 3 yıldaki trafik cezası", examples=[1])
    sehir: str = Field(..., description="Kullanım şehri", examples=["İstanbul"])
    yillik_km: int = Field(..., ge=1000, le=200000, description="Yıllık tahmini km", examples=[15000])
    garaj_var: bool = Field(..., description="Araç garajda mı korunuyor?", examples=[True])

    @field_validator("arac_tipi")
    @classmethod
    def validate_arac_tipi(cls, v):
        gecerli = ["Sedan", "SUV", "Hatchback", "Pickup", "Minivan"]
        if v not in gecerli:
            raise ValueError(f"Geçerli araç tipleri: {gecerli}")
        return v

    @field_validator("marka")
    @classmethod
    def validate_marka(cls, v):
        gecerli = ["Toyota", "Ford", "Renault", "Volkswagen", "Fiat", "Honda", "Hyundai", "BMW", "Mercedes", "Dacia"]
        if v not in gecerli:
            raise ValueError(f"Geçerli markalar: {gecerli}")
        return v

    @field_validator("sehir")
    @classmethod
    def validate_sehir(cls, v):
        gecerli = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya", "Gaziantep"]
        if v not in gecerli:
            raise ValueError(f"Geçerli şehirler: {gecerli}")
        return v


def risk_seviyesi(skor: float) -> dict:
    if skor <= 25:
        return {"seviye": "Düşük", "renk": "yeşil", "prim_carpani": 1.0, "tavsiye": "Standart prim uygulayın."}
    elif skor <= 50:
        return {"seviye": "Orta", "renk": "sarı", "prim_carpani": 1.25, "tavsiye": "%15–30 prim artışı öneririz."}
    elif skor <= 75:
        return {"seviye": "Yüksek", "renk": "turuncu", "prim_carpani": 1.50, "tavsiye": "%30–60 prim artışı veya ek teminat şartı öneririz."}
    else:
        return {"seviye": "Çok Yüksek", "renk": "kırmızı", "prim_carpani": 2.0, "tavsiye": "Poliçe reddi veya %60+ prim artışı öneririz."}


def en_riskli_faktorler(istek: RiskIstegi, skor: float) -> list:
    faktorler = []
    if istek.surucu_yasi < 25:
        faktorler.append("Genç sürücü (25 yaş altı)")
    if istek.kaza_gecmisi >= 2:
        faktorler.append(f"Yüksek kaza geçmişi ({istek.kaza_gecmisi} kaza)")
    if istek.arac_yasi >= 10:
        faktorler.append(f"Eski araç ({istek.arac_yasi} yaşında)")
    if istek.yillik_km > 30000:
        faktorler.append(f"Yüksek km ({istek.yillik_km:,} km/yıl)")
    if istek.sehir == "İstanbul":
        faktorler.append("Yüksek trafik yoğunluklu şehir")
    if istek.trafik_cezasi >= 3:
        faktorler.append(f"Çok sayıda trafik cezası ({istek.trafik_cezasi})")
    if not istek.garaj_var:
        faktorler.append("Araç açık alanda bekliyor")
    return faktorler[:4]


@app.post("/api/risk-skoru", summary="Araç risk skoru hesapla")
def risk_skoru_hesapla(istek: RiskIstegi):
    """
    Araç ve sürücü bilgilerine göre ML tabanlı risk skoru hesaplar.

    **Dönüş değerleri:**
    - `skor`: 0–100 arası risk puanı (düşük = iyi)
    - `seviye`: Düşük / Orta / Yüksek / Çok Yüksek
    - `prim_carpani`: Baz prime çarpılacak katsayı
    - `risk_faktorleri`: Skoru yükselten başlıca etkenler
    """
    X = np.array([[
        le_tip.transform([istek.arac_tipi])[0],
        le_marka.transform([istek.marka])[0],
        istek.arac_yasi,
        istek.motor_hacmi,
        istek.surucu_yasi,
        istek.surucu_deneyim,
        istek.kaza_gecmisi,
        istek.trafik_cezasi,
        le_sehir.transform([istek.sehir])[0],
        istek.yillik_km,
        int(istek.garaj_var)
    ]])

    skor = float(np.clip(model.predict(X)[0], 0, 100))
    skor_yuvarla = round(skor, 1)
    seviye_bilgi = risk_seviyesi(skor_yuvarla)
    faktorler = en_riskli_faktorler(istek, skor_yuvarla)

    return {
        "skor": skor_yuvarla,
        "seviye": seviye_bilgi["seviye"],
        "renk_kodu": seviye_bilgi["renk"],
        "prim_carpani": seviye_bilgi["prim_carpani"],
        "tavsiye": seviye_bilgi["tavsiye"],
        "risk_faktorleri": faktorler,
        "hesaplama_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_versiyonu": "1.0.0"
    }


@app.post("/api/risk-karsilastir", summary="İki aracı karşılaştır")
def risk_karsilastir(arac1: RiskIstegi, arac2: RiskIstegi):
    """İki farklı araç/sürücü profilinin risk skorunu karşılaştırır."""
    def hesapla(istek):
        X = np.array([[
            le_tip.transform([istek.arac_tipi])[0],
            le_marka.transform([istek.marka])[0],
            istek.arac_yasi, istek.motor_hacmi,
            istek.surucu_yasi, istek.surucu_deneyim,
            istek.kaza_gecmisi, istek.trafik_cezasi,
            le_sehir.transform([istek.sehir])[0],
            istek.yillik_km, int(istek.garaj_var)
        ]])
        return round(float(np.clip(model.predict(X)[0], 0, 100)), 1)

    skor1 = hesapla(arac1)
    skor2 = hesapla(arac2)

    return {
        "arac_1": {"skor": skor1, "seviye": risk_seviyesi(skor1)["seviye"]},
        "arac_2": {"skor": skor2, "seviye": risk_seviyesi(skor2)["seviye"]},
        "daha_riskli": "arac_1" if skor1 > skor2 else "arac_2",
        "fark": round(abs(skor1 - skor2), 1)
    }


@app.get("/api/referans", summary="Geçerli değerler listesi")
def referans_degerler():
    """API'ye gönderilebilecek tüm geçerli değerleri listeler."""
    return {
        "arac_tipleri": enc_data["arac_tipleri"],
        "markalar": enc_data["markalar"],
        "sehirler": enc_data["sehirler"],
        "motor_hacmi_araligi": "0.8 – 6.0 litre",
        "arac_yasi_araligi": "0 – 30 yıl",
        "surucu_yasi_araligi": "18 – 90",
        "yillik_km_araligi": "1.000 – 200.000"
    }


@app.get("/", tags=["Genel"])
def root():
    return {
        "servis": "Araç Risk Skorlama API",
        "versiyon": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /api/risk-skoru",
            "POST /api/risk-karsilastir",
            "GET  /api/referans"
        ]
    }
