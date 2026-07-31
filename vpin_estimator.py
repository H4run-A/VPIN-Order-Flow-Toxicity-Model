import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class VPINModeler:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN) Estimator
    Based on the framework by Easley, Lopez de Prado, and O'Hara.
    Adapted for high-frequency crypto tick data as a proxy for BIST ITCH data.
    """
    def __init__(self, time_bar_freq='1Min', bucket_vol_ratio=50, window_size=50):
        self.time_bar_freq = time_bar_freq
        self.bucket_vol_ratio = bucket_vol_ratio
        self.window_size = window_size
        
    def load_real_tick_data(self, file_path):
        """
        Kaggle veya Binance Vision üzerinden indirilen gerçek tick verisini yükler ve temizler.
        """
        print(f"Gerçek tick verisi yükleniyor: {file_path} ...")
        
        # CSV dosyasını okuma
        df = pd.read_csv(file_path)
        
        # Sütun isimlerini küçük harfe çevirip boşlukları temizleme
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        
        # Zaman damgasını (timestamp) ayarlama
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'time' in df.columns:
            time_col = 'time'
        elif 'date' in df.columns:
            time_col = 'date'
        else:
            raise ValueError("Zaman sütunu bulunamadı. Lütfen CSV'deki zaman sütununun adını kontrol et.")

        # Unix timestamp ise datetime'a çevir
        if str(df[time_col].dtype) in ['int64', 'float64']:
            df[time_col] = pd.to_datetime(df[time_col], unit='ms')
        else:
            df[time_col] = pd.to_datetime(df[time_col])
            
        df.set_index(time_col, inplace=True)
        
        # Fiyat sütununu bulma
        price_col = 'price' if 'price' in df.columns else 'p'
        
        # Hacim sütununu bulma (Kaggle'daki 'quantity' buraya eklendi)
        if 'quantity' in df.columns:
            volume_col = 'quantity'
        elif 'amount' in df.columns:
            volume_col = 'amount'
        elif 'qty' in df.columns:
            volume_col = 'qty'
        elif 'volume' in df.columns:
            volume_col = 'volume'
        else:
            volume_col = 'q'
        
        # Bulunan sütunları modelin beklediği 'Price' ve 'Volume' isimlerine çevirme
        df = df.rename(columns={price_col: 'Price', volume_col: 'Volume'})
        
        # Veriyi zaman sırasına göre diz ve sadece gerekli sütunları tut
        df = df.sort_index()
        df = df[['Price', 'Volume']]
        
        print(f"Toplam {len(df)} adet işlem (tick) başarıyla yüklendi.")
        return df

    def compute_time_bars(self, tick_data):
        """
        Tick verisini zaman barlarına çevirir (Toplu Hacim Sınıflandırması - Adım 1).
        """
        print(f"Tick verisi {self.time_bar_freq} zaman barlarına dönüştürülüyor...")
        time_bars = tick_data.resample(self.time_bar_freq).agg({
            'Price': ['first', 'max', 'min', 'last'],
            'Volume': 'sum'
        }).dropna()
        
        time_bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return time_bars

    def bulk_volume_classification(self, time_bars):
        """
        Fiyat değişimlerini kullanarak hacmi 'Alış' (Buy) ve 'Satış' (Sell) olarak sınıflandırır.
        """
        print("Toplu Hacim Sınıflandırması (Bulk Volume Classification) uygulanıyor...")
        time_bars['Price_Change'] = time_bars['Close'].diff()
        sigma_dp = time_bars['Price_Change'].std()
        
        # Z-skoru hesaplama
        time_bars['Z_Score'] = time_bars['Price_Change'] / sigma_dp
        
        # Normal dağılım CDF'i
        time_bars['Buy_Prob'] = norm.cdf(time_bars['Z_Score'])
        
        # Hacim dağıtımı
        time_bars['Buy_Volume'] = time_bars['Volume'] * time_bars['Buy_Prob']
        time_bars['Sell_Volume'] = time_bars['Volume'] * (1 - time_bars['Buy_Prob'])
        
        return time_bars.dropna()

    def calculate_vpin(self, classified_bars):
        """
        Eşit hacimli kovalar (equal-volume buckets) üzerinden VPIN hesaplar.
        """
        print("Eşit hacimli kovalar üzerinden VPIN hesaplanıyor...")
        
        total_days = len(np.unique(classified_bars.index.date))
        total_days = total_days if total_days > 0 else 1
        avg_daily_vol = classified_bars['Volume'].sum() / total_days
        bucket_size = avg_daily_vol / self.bucket_vol_ratio
        
        buckets = []
        current_vol = 0
        current_buy_vol = 0
        current_sell_vol = 0
        
        for index, row in classified_bars.iterrows():
            current_vol += row['Volume']
            current_buy_vol += row['Buy_Volume']
            current_sell_vol += row['Sell_Volume']
            
            while current_vol >= bucket_size:
                excess_vol = current_vol - bucket_size
                excess_ratio = excess_vol / row['Volume'] if row['Volume'] > 0 else 0
                
                buy_excess = row['Buy_Volume'] * excess_ratio
                sell_excess = row['Sell_Volume'] * excess_ratio
                
                bucket_buy = current_buy_vol - buy_excess
                bucket_sell = current_sell_vol - sell_excess
                
                buckets.append({
                    'Time': index,
                    'Buy_Volume': bucket_buy,
                    'Sell_Volume': bucket_sell,
                    'Imbalance': abs(bucket_buy - bucket_sell)
                })
                
                current_vol = excess_vol
                current_buy_vol = buy_excess
                current_sell_vol = sell_excess

        buckets_df = pd.DataFrame(buckets).set_index('Time')
        
        # Yuvarlanan (Rolling) VPIN hesaplaması
        buckets_df['VPIN'] = buckets_df['Imbalance'].rolling(window=self.window_size).sum() / (self.window_size * bucket_size)
        
        return buckets_df.dropna()

    def plot_results(self, tick_data, vpin_data):
        """
        Fiyat ve VPIN metriklerini görselleştirir.
        """
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Zaman')
        ax1.set_ylabel('Fiyat', color=color)
        ax1.plot(tick_data.index, tick_data['Price'], color=color, alpha=0.6, label='Varlık Fiyatı')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('VPIN', color=color)
        ax2.plot(vpin_data.index, vpin_data['VPIN'], color=color, linewidth=2, label='VPIN Metriği')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Kritik toksisite eşiği
        ax2.axhline(y=0.35, color='black', linestyle='--', label='Kritik Toksisite Eşiği')

        fig.tight_layout()
        plt.title('HFT Piyasa Mikro Yapısı: Fiyat vs Emir Akışı Toksisitesi (VPIN)')
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    # Modeli Başlat
    modeler = VPINModeler(time_bar_freq='1Min', bucket_vol_ratio=50, window_size=50)
    
    # Dosyanın tam yolu
    veri_dosyasi = r"C:\Users\Harun\Desktop\project1\BTCUSDT.csv" 
    
    try:
        # Veri hattını çalıştır
        tick_df = modeler.load_real_tick_data(veri_dosyasi)
        time_bars_df = modeler.compute_time_bars(tick_df)
        classified_df = modeler.bulk_volume_classification(time_bars_df)
        vpin_df = modeler.calculate_vpin(classified_df)
        
        print("İşlem tamamlandı. Sonuçlar çizdiriliyor...")
        modeler.plot_results(tick_df, vpin_df)
    
    except FileNotFoundError:
        print(f"HATA: '{veri_dosyasi}' konumu bulunamadı.")
        print("Lütfen dosya adının BTCUSDT.csv.csv olmadığına emin ol.")