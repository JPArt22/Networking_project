# Sistema de Gestión de Red Universitaria con Inteligencia Artificial

**Universidad Nacional de Colombia**  
Proyecto Integrador - Redes de Computadores

---

## 📋 Descripción

Sistema completo de gestión y monitoreo de red universitaria que integra:

- ✅ **Monitoreo en tiempo real** de métricas de red (throughput, latencia, jitter, errores)
- ✅ **Detección de anomalías** con Machine Learning (Isolation Forest)
- ✅ **Predicción de fallos** con Deep Learning (LSTM)
- ✅ **Dashboard web interactivo** con visualizaciones en tiempo real
- ✅ **Sistema de alertas** automático para eventos críticos
- ✅ **Gestión de QoS** y priorización de tráfico
- ✅ **Soporte para IEEE 802.3 (Ethernet)** e **IEEE 802.11 (WiFi)**

---

## 🏗️ Arquitectura del Sistema

```
university-network-manager/
│
├── main.py                          # Aplicación principal
├── config.yaml                      # Configuración del sistema
├── requirements.txt                 # Dependencias Python
├── README.md                        # Esta documentación
│
├── modules/                         # Módulos principales
│   ├── __init__.py
│   ├── network_monitor.py          # Captura de métricas
│   ├── anomaly_detector.py         # Detección de anomalías (IA)
│   ├── failure_predictor.py        # Predicción de fallos (LSTM)
│   ├── optimizer.py                # Optimización automática
│   └── report_generator.py         # Generación de reportes
│
├── web_dashboard/                   # Dashboard web
│   ├── __init__.py
│   ├── app.py                      # Servidor Flask
│   ├── templates/
│   │   └── dashboard.html          # Interfaz HTML
│   └── static/
│       ├── css/
│       └── js/
│
├── data/                           # Datos del sistema
│   ├── models/                     # Modelos de IA entrenados
│   ├── logs/                       # Logs del sistema
│   └── captures/                   # Capturas PCAP
│
└── tests/                          # Tests unitarios
    ├── __init__.py
    ├── test_monitor.py
    └── test_anomaly.py
```

---

## 🚀 Instalación

### Requisitos Previos

- **Python 3.8+**
- **pip** (gestor de paquetes)
- **Privilegios de administrador** (para captura de paquetes)
- **libpcap** (Linux) o **Npcap** (Windows)

### Instalación en Linux/MacOS

```bash
# Clonar o descargar el proyecto
cd university-network-manager

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios necesarios
mkdir -p data/models data/logs data/captures
```

### Instalación en Windows

```cmd
# Descargar e instalar Npcap desde: https://npcap.com/

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios
mkdir data\models data\logs data\captures
```

---

## ⚙️ Configuración

Edite el archivo `config.yaml` para ajustar parámetros:

```yaml
network:
  interfaces:
    - eth0      # Cambiar según su interfaz
    - wlan0
  
  vlans:
    10:
      name: "LABORATORIOS"
      network: "10.1.0.0/16"
    20:
      name: "WIFI_COMUN"
      network: "10.2.0.0/16"
    # ... más VLANs

monitoring:
  capture_interval: 5        # Segundos entre capturas
  packet_count: 1000         # Paquetes por captura

ai_detection:
  anomaly_model:
    contamination: 0.05      # Tasa esperada de anomalías (5%)
  
  thresholds:
    anomaly_score: -0.5      #