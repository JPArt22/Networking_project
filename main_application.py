"""
Sistema de Gestión de Red Universitaria con IA
Universidad Nacional de Colombia

Aplicación principal que integra todos los módulos:
- Monitoreo de red en tiempo real
- Detección de anomalías con IA
- Predicción de fallos con LSTM
- Dashboard web interactivo
"""

import sys
import os
import yaml
import logging
import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Any

# Añadir módulos al path
sys.path.insert(0, str(Path(__file__).parent))

# Imports de módulos del proyecto
try:
    from modules.network_monitor import NetworkMonitor
    from modules.anomaly_detector import AnomalyDetector
    from modules.failure_predictor import FailurePredictor
    from web_dashboard.app import run_dashboard, update_metrics, add_anomaly_alert, add_failure_prediction, add_log
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrese de que todos los módulos estén en la carpeta correcta")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/system.log')
    ]
)
logger = logging.getLogger(__name__)


class UniversityNetworkManager:
    """Gestor principal del sistema de red universitaria"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Inicializa el gestor de red
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        logger.info("Inicializando Sistema de Gestión de Red Universitaria")
        
        # Cargar configuración
        self.config = self.load_config(config_path)
        
        # Crear directorios necesarios
        self.create_directories()
        
        # Inicializar módulos
        self.monitor = NetworkMonitor(self.config)
        self.anomaly_detector = AnomalyDetector(
            self.config,
            model_path='data/models/anomaly_detector.pkl'
        )
        self.failure_predictor = FailurePredictor(
            self.config,
            model_path='data/models/failure_predictor.h5'
        )
        
        # Control de threads
        self.running = False
        self.monitoring_thread = None
        
        logger.info("Sistema inicializado correctamente")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            logger.info("Usando configuración por defecto")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Retorna configuración por defecto"""
        return {
            'network': {
                'interfaces': ['eth0', 'wlan0'],
                'vlans': {
                    10: {'name': 'LABORATORIOS', 'network': '10.1.0.0/16'},
                    20: {'name': 'WIFI_COMUN', 'network': '10.2.0.0/16'},
                    30: {'name': 'SERVIDORES', 'network': '10.3.0.0/24'},
                    40: {'name': 'ADMINISTRACION', 'network': '10.4.0.0/22'},
                    50: {'name': 'GESTION', 'network': '10.5.0.0/24'},
                    60: {'name': 'DMZ', 'network': '10.6.0.0/24'},
                    70: {'name': 'VOIP', 'network': '10.7.0.0/20'}
                }
            },
            'monitoring': {
                'capture_interval': 5,
                'packet_count': 1000
            },
            'ai_detection': {
                'anomaly_model': {
                    'contamination': 0.05,
                    'random_state': 42
                },
                'failure_model': {
                    'sequence_length': 24,
                    'prediction_hours': 2
                },
                'thresholds': {
                    'anomaly_score': -0.5,
                    'failure_probability': 0.7
                }
            },
            'dashboard': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            }
        }
    
    def create_directories(self):
        """Crea directorios necesarios si no existen"""
        directories = [
            'data/models',
            'data/logs',
            'data/captures'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directorio verificado: {directory}")
    
    def monitoring_loop(self):
        """Loop principal de monitoreo"""
        logger.info("Iniciando loop de monitoreo")
        
        metrics_history = []
        capture_count = 0
        
        while self.running:
            try:
                # Capturar métricas
                logger.info(f"Captura #{capture_count + 1}")
                
                metrics = self.monitor.start_monitoring(
                    duration=self.config['monitoring']['capture_interval']
                )
                
                if metrics:
                    # Actualizar dashboard
                    update_metrics(metrics)
                    
                    # Guardar en historial
                    metrics_history.append(metrics)
                    
                    # Mantener solo últimas 100 capturas
                    if len(metrics_history) > 100:
                        metrics_history = metrics_history[-100:]
                    
                    # Entrenar modelos si es necesario
                    if capture_count == 10 and not self.anomaly_detector.is_trained:
                        logger.info("Entrenando detector de anomalías...")
                        self.anomaly_detector.train(metrics_history)
                        self.anomaly_detector.save_model('data/models/anomaly_detector.pkl')
                    
                    # Detección de anomalías
                    if self.anomaly_detector.is_trained:
                        anomaly_result = self.anomaly_detector.detect(metrics)
                        
                        if anomaly_result['is_anomaly']:
                            add_anomaly_alert(anomaly_result)
                            logger.warning(f"Anomalía detectada: {anomaly_result['message']}")
                    
                    # Predicción de fallos (solo si hay suficiente historial)
                    if len(metrics_history) >= 24:
                        if not self.failure_predictor.is_trained:
                            # Entrenar con datos sintéticos para demostración
                            logger.info("Predictor de fallos requiere entrenamiento previo")
                        else:
                            prediction = self.failure_predictor.predict(metrics_history)
                            
                            if prediction['prediction']:
                                add_failure_prediction(prediction)
                                logger.error(f"Fallo predicho: {prediction['message']}")
                    
                    capture_count += 1
                    
                    # Log de estado
                    if capture_count % 10 == 0:
                        logger.info(f"Capturas completadas: {capture_count}")
                        stats = self.anomaly_detector.get_detection_statistics()
                        if 'error' not in stats:
                            logger.info(f"Anomalías detectadas: {stats['anomalies_detected']}/{stats['total_detections']}")
                
                else:
                    logger.warning("No se pudieron capturar métricas")
                
            except KeyboardInterrupt:
                logger.info("Interrupción recibida")
                break
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}", exc_info=True)
                time.sleep(5)  # Esperar antes de reintentar
        
        logger.info("Loop de monitoreo finalizado")
    
    def start(self, mode: str = 'full'):
        """
        Inicia el sistema
        
        Args:
            mode: Modo de operación
                  - 'full': Monitoreo + Dashboard
                  - 'monitor': Solo monitoreo
                  - 'dashboard': Solo dashboard (con simulación)
        """
        logger.info(f"Iniciando sistema en modo: {mode}")
        
        if mode in ['full', 'monitor']:
            # Iniciar monitoreo en thread separado
            self.running = True
            self.monitoring_thread = threading.Thread(
                target=self.monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Thread de monitoreo iniciado")
        
        if mode in ['full', 'dashboard']:
            # Iniciar dashboard (bloquea hasta Ctrl+C)
            simulation = (mode == 'dashboard')
            logger.info(f"Iniciando dashboard (simulación: {simulation})")
            run_dashboard(self.config, simulation=simulation)
        
        # Si solo es monitor, mantener vivo
        if mode == 'monitor':
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Deteniendo sistema...")
                self.stop()
    
    def stop(self):
        """Detiene el sistema"""
        logger.info("Deteniendo sistema...")
        self.running = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Guardar modelos
        if self.anomaly_detector.is_trained:
            self.anomaly_detector.save_model('data/models/anomaly_detector.pkl')
        
        logger.info("Sistema detenido")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Gestión de Red Universitaria con IA'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ruta al archivo de configuración (default: config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'monitor', 'dashboard'],
        default='full',
        help='Modo de operación (default: full)'
    )
    
    parser.add_argument(
        '--train-models',
        action='store_true',
        help='Entrenar modelos de IA con datos sintéticos'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 80)
    print("SISTEMA DE GESTIÓN DE RED UNIVERSITARIA CON INTELIGENCIA ARTIFICIAL")
    print("Universidad Nacional de Colombia")
    print("=" * 80)
    print()
    
    try:
        # Crear gestor
        manager = UniversityNetworkManager(args.config)
        
        # Entrenar modelos si se solicita
        if args.train_models:
            print("Entrenando modelos de IA...")
            train_models(manager)
            print("Modelos entrenados y guardados")
            return
        
        # Iniciar sistema
        print(f"Iniciando en modo: {args.mode}")
        print()
        
        if args.mode == 'full':
            print("Dashboard disponible en: http://localhost:5000")
            print("Presione Ctrl+C para detener")
        elif args.mode == 'dashboard':
            print("Dashboard en modo simulación")
            print("Disponible en: http://localhost:5000")
            print("Presione Ctrl+C para detener")
        elif args.mode == 'monitor':
            print("Modo solo monitoreo")
            print("Los datos se capturarán pero no habrá interfaz web")
            print("Presione Ctrl+C para detener")
        
        print()
        
        manager.start(mode=args.mode)
        
    except KeyboardInterrupt:
        print("\n\nSistema detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        sys.exit(1)


def train_models(manager: UniversityNetworkManager):
    """Entrena modelos con datos sintéticos"""
    import numpy as np
    
    logger.info("Generando datos sintéticos para entrenamiento...")
    
    # Generar datos para detector de anomalías
    training_metrics = []
    for _ in range(50):
        metrics = {
            'total_packets': np.random.randint(800, 1200),
            'capture_duration': 5.0,
            'latency': {
                'avg_ms': np.random.uniform(10, 30),
                'jitter_ms': np.random.uniform(1, 5),
                'max_ms': np.random.uniform(30, 60)
            },
            'system': {
                'cpu_percent': np.random.uniform(20, 60),
                'memory_percent': np.random.uniform(40, 70),
                'disk_percent': np.random.uniform(30, 60),
                'network_connections': np.random.randint(50, 150)
            },
            'vlans': {
                10: {
                    'throughput_mbps': np.random.uniform(100, 300),
                    'error_rate': np.random.uniform(0, 2),
                    'protocol_distribution': {
                        'TCP': np.random.uniform(40, 60),
                        'UDP': np.random.uniform(20, 40),
                        'ICMP': np.random.uniform(5, 15)
                    }
                }
            }
        }
        training_metrics.append(metrics)
    
    # Entrenar detector de anomalías
    logger.info("Entrenando detector de anomalías...")
    manager.anomaly_detector.train(training_metrics)
    manager.anomaly_detector.save_model('data/models/anomaly_detector.pkl')
    logger.info("Detector de anomalías entrenado")
    
    # Nota: El predictor de fallos requiere datos etiquetados
    logger.info("Nota: El predictor de fallos requiere datos históricos reales etiquetados")
    logger.info("Se debe entrenar manualmente con datos de producción")


if __name__ == "__main__":
    main()