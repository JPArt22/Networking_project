# ==============================================================================
# ESTRUCTURA DEL PROYECTO
# ==============================================================================
"""
university-network-manager/
│
├── main.py                          # Punto de entrada principal
├── requirements.txt                 # Dependencias
├── config.yaml                      # Configuración
├── README.md                        # Documentación
│
├── modules/
│   ├── __init__.py
│   ├── network_monitor.py          # Captura de métricas en tiempo real
│   ├── anomaly_detector.py         # Detección de anomalías con IA
│   ├── failure_predictor.py        # Predicción de fallos (LSTM)
│   ├── optimizer.py                # Optimización automática de red
│   └── report_generator.py         # Generador de reportes
│
├── web_dashboard/
│   ├── __init__.py
│   ├── app.py                      # Aplicación Flask
│   ├── templates/
│   │   └── dashboard.html          # Dashboard HTML
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── dashboard.js        # Chart.js visualizaciones
│
├── data/
│   ├── models/                     # Modelos de IA entrenados
│   ├── logs/                       # Logs del sistema
│   └── captures/                   # Capturas PCAP
│
└── tests/
    ├── __init__.py
    ├── test_monitor.py
    └── test_anomaly.py
"""

# ==============================================================================
# requirements.txt
# ==============================================================================
REQUIREMENTS = """
# Core dependencies
scapy==2.5.0
numpy==1.24.3
pandas==2.0.3

# Machine Learning
scikit-learn==1.3.0
tensorflow==2.13.0
joblib==1.3.2

# Web Dashboard
flask==2.3.3
flask-cors==4.0.0
flask-socketio==5.3.4

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
psutil==5.9.5
"""

# ==============================================================================
# config.yaml
# ==============================================================================
CONFIG_YAML = """
# Configuración del Sistema de Gestión de Red Universitaria

network:
  # Interfaces de red a monitorear
  interfaces:
    - eth0
    - wlan0
  
  # Segmentos de red (VLANs)
  vlans:
    10:
      name: "LABORATORIOS"
      network: "10.1.0.0/16"
    20:
      name: "WIFI_COMUN"
      network: "10.2.0.0/16"
    30:
      name: "SERVIDORES"
      network: "10.3.0.0/24"
    40:
      name: "ADMINISTRACION"
      network: "10.4.0.0/22"
    50:
      name: "GESTION"
      network: "10.5.0.0/24"
    60:
      name: "DMZ"
      network: "10.6.0.0/24"
    70:
      name: "VOIP"
      network: "10.7.0.0/20"

monitoring:
  # Intervalo de captura (segundos)
  capture_interval: 5
  
  # Número de paquetes por captura
  packet_count: 1000
  
  # Métricas a recopilar
  metrics:
    - throughput
    - latency
    - packet_loss
    - jitter
    - error_rate
    - cpu_usage
    - memory_usage

ai_detection:
  # Modelo de detección de anomalías
  anomaly_model:
    type: "IsolationForest"
    contamination: 0.05
    random_state: 42
  
  # Modelo de predicción de fallos
  failure_model:
    type: "LSTM"
    sequence_length: 24
    prediction_hours: 2
  
  # Umbrales de alerta
  thresholds:
    anomaly_score: -0.5
    failure_probability: 0.7

qos:
  # Configuración de QoS
  classes:
    EF:  # Expedited Forwarding (VoIP)
      priority: 1
      bandwidth_percent: 20
      dscp: 46
    AF41:  # Assured Forwarding (Video)
      priority: 2
      bandwidth_percent: 30
      dscp: 34
    AF31:  # Assured Forwarding (Académico)
      priority: 3
      bandwidth_percent: 35
      dscp: 26
    BE:  # Best Effort (Recreativo)
      priority: 4
      bandwidth_percent: 15
      dscp: 0

dashboard:
  # Configuración del dashboard web
  host: "0.0.0.0"
  port: 5000
  debug: false
  update_interval: 5  # segundos

logging:
  level: "INFO"
  file: "data/logs/system.log"
  max_size: "10MB"
  backup_count: 5
"""

print("=" * 80)
print("ESTRUCTURA DEL PROYECTO CREADA")
print("=" * 80)
print("\nArchivos de configuración generados:")
print("  - requirements.txt")
print("  - config.yaml")
print("\nDirectorios necesarios:")
print("  - modules/")
print("  - web_dashboard/")
print("  - data/models/")
print("  - data/logs/")
print("  - data/captures/")
print("  - tests/")
print("\n" + "=" * 80)