"""
Módulo de Predicción de Fallos con Redes Neuronales LSTM
Universidad Nacional de Colombia - Sistema de Gestión de Red

Predice fallos de dispositivos y degradación de red con 2 horas de anticipación
utilizando redes neuronales recurrentes (LSTM).
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import os

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow no disponible. Predicción de fallos deshabilitada.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailurePredictor:
    """Predictor de fallos basado en LSTM"""
    
    def __init__(self, config: Dict[str, Any], model_path: str = None):
        """
        Inicializa el predictor de fallos
        
        Args:
            config: Configuración del sistema
            model_path: Ruta al modelo pre-entrenado (opcional)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow es requerido para predicción de fallos")
        
        self.config = config
        self.ai_config = config.get('ai_detection', {})
        
        # Parámetros del modelo
        failure_config = self.ai_config.get('failure_model', {})
        self.sequence_length = failure_config.get('sequence_length', 24)
        self.prediction_hours = failure_config.get('prediction_hours', 2)
        
        # Cargar o crear modelo
        if model_path and os.path.exists(model_path):
            logger.info(f"Cargando modelo LSTM desde {model_path}")
            self.model = load_model(model_path)
            self.is_trained = True
        else:
            logger.info("Creando nuevo modelo LSTM")
            self.model = self._build_model()
            self.is_trained = False
        
        # Historial de secuencias
        self.sequences = []
        
        # Umbral de probabilidad de fallo
        self.failure_threshold = self.ai_config.get('thresholds', {}).get('failure_probability', 0.7)
        
        logger.info("Predictor de fallos inicializado")
    
    def _build_model(self) -> Sequential:
        """
        Construye la arquitectura del modelo LSTM
        
        Returns:
            Modelo LSTM de Keras
        """
        model = Sequential([
            # Primera capa LSTM
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 15)),
            Dropout(0.2),
            
            # Segunda capa LSTM
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Capas densas
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Capa de salida (probabilidad de fallo)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        logger.info("Modelo LSTM construido exitosamente")
        logger.info(f"Parámetros: {model.count_params():,}")
        
        return model
    
    def prepare_sequence(self, metrics_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prepara secuencia temporal de métricas
        
        Args:
            metrics_history: Lista de métricas históricas
            
        Returns:
            Array numpy con secuencia preparada
        """
        if len(metrics_history) < self.sequence_length:
            # Pad con ceros si no hay suficientes datos
            padding = self.sequence_length - len(metrics_history)
            metrics_history = [self._create_empty_metrics()] * padding + metrics_history
        
        # Extraer últimas sequence_length métricas
        recent_metrics = metrics_history[-self.sequence_length:]
        
        # Convertir a features
        sequence = []
        for metrics in recent_metrics:
            features = self._extract_features(metrics)
            sequence.append(features)
        
        return np.array(sequence).reshape(1, self.sequence_length, -1)
    
    def _extract_features(self, metrics: Dict[str, Any]) -> List[float]:
        """
        Extrae características de una métrica
        
        Args:
            metrics: Diccionario con métricas
            
        Returns:
            Lista de características
        """
        features = []
        
        # Métricas de latencia
        latency = metrics.get('latency', {})
        features.extend([
            latency.get('avg_ms', 0),
            latency.get('jitter_ms', 0),
            latency.get('max_ms', 0)
        ])
        
        # Métricas del sistema
        system = metrics.get('system', {})
        features.extend([
            system.get('cpu_percent', 0),
            system.get('memory_percent', 0),
            system.get('disk_percent', 0),
            system.get('network_connections', 0)
        ])
        
        # Métricas de red
        features.append(metrics.get('total_packets', 0) / 1000.0)  # Normalizar
        
        # Métricas por VLAN (promedios)
        vlans = metrics.get('vlans', {})
        if vlans:
            throughputs = [v.get('throughput_mbps', 0) for v in vlans.values()]
            error_rates = [v.get('error_rate', 0) for v in vlans.values()]
            
            features.extend([
                np.mean(throughputs) if throughputs else 0,
                np.max(throughputs) if throughputs else 0,
                np.mean(error_rates) if error_rates else 0,
                np.max(error_rates) if error_rates else 0
            ])
            
            # Distribución de protocolos
            tcp_avg = np.mean([
                v.get('protocol_distribution', {}).get('TCP', 0) 
                for v in vlans.values()
            ])
            udp_avg = np.mean([
                v.get('protocol_distribution', {}).get('UDP', 0) 
                for v in vlans.values()
            ])
            icmp_avg = np.mean([
                v.get('protocol_distribution', {}).get('ICMP', 0) 
                for v in vlans.values()
            ])
            
            features.extend([tcp_avg, udp_avg, icmp_avg])
        else:
            features.extend([0] * 7)
        
        return features
    
    def _create_empty_metrics(self) -> Dict[str, Any]:
        """
        Crea un diccionario de métricas vacío
        
        Returns:
            Diccionario con valores por defecto
        """
        return {
            'latency': {'avg_ms': 0, 'jitter_ms': 0, 'max_ms': 0},
            'system': {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0, 'network_connections': 0},
            'total_packets': 0,
            'vlans': {}
        }
    
    def train(self, training_data: List[Tuple[List[Dict], int]], 
              validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32):
        """
        Entrena el modelo LSTM
        
        Args:
            training_data: Lista de tuplas (secuencia_metricas, etiqueta_fallo)
                          etiqueta_fallo: 0 = normal, 1 = fallo
            validation_split: Proporción de datos para validación
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
        """
        if len(training_data) < 20:
            logger.warning("Se necesitan al menos 20 secuencias para entrenar")
            return False
        
        logger.info(f"Entrenando modelo LSTM con {len(training_data)} secuencias")
        
        # Preparar datos
        X = []
        y = []
        
        for sequence, label in training_data:
            seq_array = self.prepare_sequence(sequence)
            X.append(seq_array[0])
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Shape de entrenamiento: X={X.shape}, y={y.shape}")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Entrenar
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.is_trained = True
        
        # Reporte de entrenamiento
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        
        logger.info(f"Entrenamiento completado:")
        logger.info(f"  Loss: {final_loss:.4f} | Accuracy: {final_accuracy:.4f}")
        logger.info(f"  Val Loss: {final_val_loss:.4f} | Val Accuracy: {final_val_accuracy:.4f}")
        
        return True
    
    def predict(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predice probabilidad de fallo
        
        Args:
            metrics_history: Historial de métricas
            
        Returns:
            Diccionario con predicción
        """
        if not self.is_trained:
            logger.warning("El modelo no ha sido entrenado")
            return {
                'failure_probability': 0.0,
                'prediction': False,
                'message': 'Modelo no entrenado',
                'timestamp': datetime.now().isoformat()
            }
        
        # Preparar secuencia
        sequence = self.prepare_sequence(metrics_history)
        
        # Predecir
        probability = float(self.model.predict(sequence, verbose=0)[0][0])
        
        # Determinar si hay riesgo de fallo
        will_fail = probability >= self.failure_threshold
        
        # Calcular tiempo estimado hasta el fallo
        eta = None
        if will_fail:
            # Estimación basada en la probabilidad
            hours_until_failure = self.prediction_hours * (1 - (probability - self.failure_threshold) / (1 - self.failure_threshold))
            eta = datetime.now() + timedelta(hours=hours_until_failure)
        
        result = {
            'failure_probability': round(probability * 100, 2),
            'prediction': will_fail,
            'severity': self._calculate_severity(probability),
            'eta': eta.isoformat() if eta else None,
            'message': self._generate_message(probability, will_fail),
            'timestamp': datetime.now().isoformat(),
            'confidence': round((abs(probability - 0.5) * 2) * 100, 2)
        }
        
        if will_fail:
            logger.warning(f"FALLO PREDICHO: {result['message']}")
        
        return result
    
    def _calculate_severity(self, probability: float) -> str:
        """Calcula severidad basada en probabilidad"""
        if probability < 0.5:
            return 'normal'
        elif probability < 0.7:
            return 'low'
        elif probability < 0.85:
            return 'medium'
        elif probability < 0.95:
            return 'high'
        else:
            return 'critical'
    
    def _generate_message(self, probability: float, will_fail: bool) -> str:
        """Genera mensaje descriptivo"""
        if not will_fail:
            return "Sistema operando normalmente"
        
        prob_percent = probability * 100
        
        if probability >= 0.95:
            return f"CRÍTICO: Fallo inminente ({prob_percent:.1f}% probabilidad) - Acción inmediata requerida"
        elif probability >= 0.85:
            return f"ALTO: Fallo probable en las próximas {self.prediction_hours} horas ({prob_percent:.1f}%)"
        else:
            return f"MEDIO: Riesgo de fallo detectado ({prob_percent:.1f}%) - Monitorear de cerca"
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            logger.warning("No se puede guardar un modelo no entrenado")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Modelo guardado en {path}")
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False


def main():
    """Función principal para pruebas"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow no está instalado. Instale con: pip install tensorflow")
        return
    
    config = {
        'ai_detection': {
            'failure_model': {
                'sequence_length': 24,
                'prediction_hours': 2
            },
            'thresholds': {
                'failure_probability': 0.7
            }
        }
    }
    
    predictor = FailurePredictor(config)
    
    # Generar datos sintéticos
    print("Generando datos de entrenamiento sintéticos...")
    
    training_data = []
    
    # Generar secuencias normales (etiqueta 0)
    for _ in range(40):
        sequence = []
        for i in range(24):
            metrics = {
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
                'total_packets': np.random.randint(800, 1200),
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
            sequence.append(metrics)
        training_data.append((sequence, 0))
    
    # Generar secuencias con fallo (etiqueta 1)
    for _ in range(10):
        sequence = []
        for i in range(24):
            # Degradación progresiva
            degradation = i / 24.0
            metrics = {
                'latency': {
                    'avg_ms': np.random.uniform(20 + degradation * 200, 40 + degradation * 200),
                    'jitter_ms': np.random.uniform(5 + degradation * 40, 10 + degradation * 40),
                    'max_ms': np.random.uniform(60 + degradation * 400, 100 + degradation * 400)
                },
                'system': {
                    'cpu_percent': np.random.uniform(60 + degradation * 30, 80 + degradation * 15),
                    'memory_percent': np.random.uniform(70 + degradation * 20, 85 + degradation * 10),
                    'disk_percent': np.random.uniform(60, 85),
                    'network_connections': np.random.randint(150, 300)
                },
                'total_packets': np.random.randint(1500, 3000),
                'vlans': {
                    10: {
                        'throughput_mbps': np.random.uniform(400 + degradation * 400, 600 + degradation * 300),
                        'error_rate': np.random.uniform(3 + degradation * 10, 8 + degradation * 10),
                        'protocol_distribution': {
                            'TCP': np.random.uniform(30, 50),
                            'UDP': np.random.uniform(15, 30),
                            'ICMP': np.random.uniform(20 + degradation * 40, 40 + degradation * 50)
                        }
                    }
                }
            }
            sequence.append(metrics)
        training_data.append((sequence, 1))
    
    # Entrenar
    print(f"\nEntrenando modelo con {len(training_data)} secuencias...")
    predictor.train(training_data, epochs=30, batch_size=16)
    
    # Probar predicción
    print("\n" + "=" * 80)
    print("PRUEBA: Predicción en secuencia normal")
    print("=" * 80)
    
    normal_sequence = training_data[0][0]
    result = predictor.predict(normal_sequence)
    print(f"Probabilidad de fallo: {result['failure_probability']}%")
    print(f"Predicción: {'FALLO' if result['prediction'] else 'NORMAL'}")
    print(f"Severidad: {result['severity']}")
    print(f"Mensaje: {result['message']}")
    
    print("\n" + "=" * 80)
    print("PRUEBA: Predicción en secuencia con fallo")
    print("=" * 80)
    
    failure_sequence = training_data[-1][0]
    result = predictor.predict(failure_sequence)
    print(f"Probabilidad de fallo: {result['failure_probability']}%")
    print(f"Predicción: {'FALLO' if result['prediction'] else 'NORMAL'}")
    print(f"Severidad: {result['severity']}")
    print(f"ETA: {result['eta']}")
    print(f"Mensaje: {result['message']}")
    
    # Guardar modelo
    print("\nGuardando modelo...")
    predictor.save_model('data/models/failure_predictor.h5')


if __name__ == "__main__":
    main()