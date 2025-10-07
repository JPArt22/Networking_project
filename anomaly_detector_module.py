"""
Módulo de Detección de Anomalías con Inteligencia Artificial
Universidad Nacional de Colombia - Sistema de Gestión de Red

Utiliza Isolation Forest para detectar comportamiento anómalo en:
- Patrones de tráfico
- Tasas de error
- Distribución de protocolos
- Utilización de recursos
"""

import numpy as np
import joblib
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detector de anomalías basado en Isolation Forest"""
    
    def __init__(self, config: Dict[str, Any], model_path: str = None):
        """
        Inicializa el detector de anomalías
        
        Args:
            config: Configuración del sistema
            model_path: Ruta al modelo pre-entrenado (opcional)
        """
        self.config = config
        self.ai_config = config.get('ai_detection', {})
        
        # Parámetros del modelo
        contamination = self.ai_config.get('anomaly_model', {}).get('contamination', 0.05)
        random_state = self.ai_config.get('anomaly_model', {}).get('random_state', 42)
        
        # Inicializar modelo
        if model_path and os.path.exists(model_path):
            logger.info(f"Cargando modelo desde {model_path}")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
            self.is_trained = True
        else:
            logger.info("Creando nuevo modelo Isolation Forest")
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1
            )
            self.scaler = StandardScaler()
            self.is_trained = False
        
        # Historial de detecciones
        self.detection_history = []
        
        # Umbrales
        self.anomaly_threshold = self.ai_config.get('thresholds', {}).get('anomaly_score', -0.5)
        
        logger.info("Detector de anomalías inicializado")
    
    def extract_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        """
        Extrae características relevantes de las métricas
        
        Args:
            metrics: Diccionario con métricas de red
            
        Returns:
            Array numpy con características extraídas
        """
        features = []
        
        # Métricas globales
        features.append(metrics.get('total_packets', 0))
        features.append(metrics.get('capture_duration', 0))
        
        # Métricas de latencia
        latency = metrics.get('latency', {})
        features.append(latency.get('avg_ms', 0))
        features.append(latency.get('jitter_ms', 0))
        features.append(latency.get('max_ms', 0))
        
        # Métricas del sistema
        system = metrics.get('system', {})
        features.append(system.get('cpu_percent', 0))
        features.append(system.get('memory_percent', 0))
        features.append(system.get('network_connections', 0))
        
        # Métricas por VLAN (promedios)
        vlans = metrics.get('vlans', {})
        if vlans:
            throughputs = [v.get('throughput_mbps', 0) for v in vlans.values()]
            error_rates = [v.get('error_rate', 0) for v in vlans.values()]
            
            features.append(np.mean(throughputs) if throughputs else 0)
            features.append(np.max(throughputs) if throughputs else 0)
            features.append(np.std(throughputs) if len(throughputs) > 1 else 0)
            
            features.append(np.mean(error_rates) if error_rates else 0)
            features.append(np.max(error_rates) if error_rates else 0)
            
            # Distribución de protocolos (promedio de TCP, UDP, ICMP)
            tcp_percent = []
            udp_percent = []
            icmp_percent = []
            
            for vlan_data in vlans.values():
                proto_dist = vlan_data.get('protocol_distribution', {})
                tcp_percent.append(proto_dist.get('TCP', 0))
                udp_percent.append(proto_dist.get('UDP', 0))
                icmp_percent.append(proto_dist.get('ICMP', 0))
            
            features.append(np.mean(tcp_percent) if tcp_percent else 0)
            features.append(np.mean(udp_percent) if udp_percent else 0)
            features.append(np.mean(icmp_percent) if icmp_percent else 0)
        else:
            # Añadir ceros si no hay datos de VLAN
            features.extend([0] * 10)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, metrics_history: List[Dict[str, Any]]):
        """
        Entrena el modelo con datos históricos
        
        Args:
            metrics_history: Lista de métricas históricas
        """
        if len(metrics_history) < 10:
            logger.warning("Se necesitan al menos 10 muestras para entrenar el modelo")
            return False
        
        logger.info(f"Entrenando modelo con {len(metrics_history)} muestras")
        
        # Extraer características de todas las muestras
        features_list = []
        for metrics in metrics_history:
            features = self.extract_features(metrics)
            features_list.append(features[0])
        
        X = np.array(features_list)
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar modelo
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info("Modelo entrenado exitosamente")
        return True
    
    def detect(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta anomalías en las métricas proporcionadas
        
        Args:
            metrics: Diccionario con métricas actuales
            
        Returns:
            Diccionario con resultado de detección
        """
        if not self.is_trained:
            logger.warning("El modelo no ha sido entrenado. Entrene primero con datos históricos.")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'message': 'Modelo no entrenado',
                'timestamp': datetime.now().isoformat()
            }
        
        # Extraer características
        features = self.extract_features(metrics)
        
        # Normalizar
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        # Determinar si es anomalía
        is_anomaly = prediction == -1 or anomaly_score < self.anomaly_threshold
        
        # Calcular confianza (0-100%)
        confidence = min(100, max(0, abs(anomaly_score) * 100))
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat(),
            'features': features[0].tolist(),
            'thresholds_exceeded': self._check_thresholds(metrics)
        }
        
        # Añadir detalles si es anomalía
        if is_anomaly:
            result['severity'] = self._calculate_severity(anomaly_score)
            result['message'] = self._generate_anomaly_message(metrics, result)
            logger.warning(f"ANOMALÍA DETECTADA: {result['message']}")
        else:
            result['severity'] = 'normal'
            result['message'] = 'Comportamiento normal'
        
        # Guardar en historial
        self.detection_history.append(result)
        
        return result
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Verifica si alguna métrica excede umbrales definidos
        
        Args:
            metrics: Métricas actuales
            
        Returns:
            Lista de umbrales excedidos
        """
        exceeded = []
        
        # Verificar CPU
        cpu = metrics.get('system', {}).get('cpu_percent', 0)
        if cpu > 80:
            exceeded.append(f"CPU alta: {cpu}%")
        
        # Verificar memoria
        memory = metrics.get('system', {}).get('memory_percent', 0)
        if memory > 85:
            exceeded.append(f"Memoria alta: {memory}%")
        
        # Verificar latencia
        latency = metrics.get('latency', {}).get('avg_ms', 0)
        if latency > 100:
            exceeded.append(f"Latencia alta: {latency} ms")
        
        # Verificar jitter
        jitter = metrics.get('latency', {}).get('jitter_ms', 0)
        if jitter > 20:
            exceeded.append(f"Jitter alto: {jitter} ms")
        
        # Verificar error rate por VLAN
        for vlan_id, vlan_data in metrics.get('vlans', {}).items():
            error_rate = vlan_data.get('error_rate', 0)
            if error_rate > 5:
                vlan_name = vlan_data.get('name', f'VLAN_{vlan_id}')
                exceeded.append(f"Error rate alto en {vlan_name}: {error_rate}%")
        
        return exceeded
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """
        Calcula la severidad de la anomalía
        
        Args:
            anomaly_score: Score de anomalía del modelo
            
        Returns:
            Nivel de severidad: 'low', 'medium', 'high', 'critical'
        """
        if anomaly_score > -0.3:
            return 'low'
        elif anomaly_score > -0.5:
            return 'medium'
        elif anomaly_score > -0.7:
            return 'high'
        else:
            return 'critical'
    
    def _generate_anomaly_message(self, metrics: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Genera mensaje descriptivo de la anomalía
        
        Args:
            metrics: Métricas actuales
            result: Resultado de detección
            
        Returns:
            Mensaje descriptivo
        """
        severity = result.get('severity', 'unknown')
        thresholds = result.get('thresholds_exceeded', [])
        
        message = f"Anomalía de severidad {severity.upper()} detectada"
        
        if thresholds:
            message += f" - Umbrales excedidos: {', '.join(thresholds)}"
        
        # Añadir detalles específicos
        details = []
        
        # Verificar latencia anormal
        latency = metrics.get('latency', {}).get('avg_ms', 0)
        if latency > 150:
            details.append(f"latencia crítica ({latency} ms)")
        
        # Verificar throughput anormal
        vlans = metrics.get('vlans', {})
        for vlan_data in vlans.values():
            throughput = vlan_data.get('throughput_mbps', 0)
            if throughput > 800:  # Throughput inusualmente alto
                details.append(f"throughput alto en {vlan_data.get('name')}")
        
        if details:
            message += f" - Detalles: {', '.join(details)}"
        
        return message
    
    def save_model(self, path: str):
        """
        Guarda el modelo entrenado
        
        Args:
            path: Ruta donde guardar el modelo
        """
        if not self.is_trained:
            logger.warning("No se puede guardar un modelo no entrenado")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            joblib.dump(self.scaler, path.replace('.pkl', '_scaler.pkl'))
            logger.info(f"Modelo guardado en {path}")
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de detecciones
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.detection_history:
            return {'error': 'No hay detecciones en el historial'}
        
        total = len(self.detection_history)
        anomalies = sum(1 for d in self.detection_history if d['is_anomaly'])
        
        # Contar por severidad
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for detection in self.detection_history:
            if detection['is_anomaly']:
                severity = detection.get('severity', 'low')
                severity_counts[severity] += 1
        
        return {
            'total_detections': total,
            'anomalies_detected': anomalies,
            'anomaly_rate': round((anomalies / total) * 100, 2),
            'severity_distribution': severity_counts,
            'avg_confidence': round(
                np.mean([d['confidence'] for d in self.detection_history]), 2
            )
        }


def main():
    """Función principal para pruebas"""
    # Configuración de prueba
    config = {
        'ai_detection': {
            'anomaly_model': {
                'contamination': 0.05,
                'random_state': 42
            },
            'thresholds': {
                'anomaly_score': -0.5
            }
        }
    }
    
    # Crear detector
    detector = AnomalyDetector(config)
    
    # Generar datos de entrenamiento sintéticos (simulando métricas normales)
    print("Generando datos de entrenamiento...")
    training_data = []
    for i in range(50):
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
        training_data.append(metrics)
    
    # Entrenar modelo
    print("Entrenando modelo...")
    detector.train(training_data)
    
    # Probar con datos normales
    print("\n" + "=" * 80)
    print("PRUEBA 1: Datos normales")
    print("=" * 80)
    normal_metrics = training_data[0]
    result = detector.detect(normal_metrics)
    print(f"Anomalía detectada: {result['is_anomaly']}")
    print(f"Score: {result['anomaly_score']:.4f}")
    print(f"Confianza: {result['confidence']:.2f}%")
    print(f"Mensaje: {result['message']}")
    
    # Probar con datos anómalos
    print("\n" + "=" * 80)
    print("PRUEBA 2: Datos anómalos (latencia alta)")
    print("=" * 80)
    anomalous_metrics = {
        'total_packets': 2500,  # Inusualmente alto
        'capture_duration': 5.0,
        'latency': {
            'avg_ms': 250,  # Latencia muy alta
            'jitter_ms': 45,  # Jitter muy alto
            'max_ms': 500
        },
        'system': {
            'cpu_percent': 95,  # CPU saturada
            'memory_percent': 88,
            'network_connections': 300
        },
        'vlans': {
            10: {
                'throughput_mbps': 850,  # Throughput muy alto
                'error_rate': 8,  # Error rate alto
                'protocol_distribution': {
                    'TCP': 30,
                    'UDP': 10,
                    'ICMP': 60  # ICMP inusualmente alto (posible ataque)
                }
            }
        }
    }
    result = detector.detect(anomalous_metrics)
    print(f"Anomalía detectada: {result['is_anomaly']}")
    print(f"Score: {result['anomaly_score']:.4f}")
    print(f"Confianza: {result['confidence']:.2f}%")
    print(f"Severidad: {result['severity']}")
    print(f"Mensaje: {result['message']}")
    print(f"Umbrales excedidos: {result['thresholds_exceeded']}")
    
    # Estadísticas
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE DETECCIÓN")
    print("=" * 80)
    stats = detector.get_detection_statistics()
    print(f"Total de detecciones: {stats['total_detections']}")
    print(f"Anomalías detectadas: {stats['anomalies_detected']}")
    print(f"Tasa de anomalías: {stats['anomaly_rate']}%")
    print(f"Distribución por severidad: {stats['severity_distribution']}")
    
    # Guardar modelo
    print("\nGuardando modelo...")
    detector.save_model('data/models/anomaly_detector.pkl')


if __name__ == "__main__":
    main()