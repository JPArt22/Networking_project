"""
Dashboard Web para Gestión de Red Universitaria
Universidad Nacional de Colombia

Dashboard en tiempo real con Flask y WebSockets
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'universidad-nacional-colombia-2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Variables globales para almacenar estado
current_metrics = {}
anomaly_alerts = []
failure_predictions = []
system_logs = []

# Lock para thread-safety
data_lock = threading.Lock()


@app.route('/')
def index():
    """Página principal del dashboard"""
    return render_template('dashboard.html')


@app.route('/api/metrics/current')
def get_current_metrics():
    """Obtiene métricas actuales"""
    with data_lock:
        return jsonify(current_metrics)


@app.route('/api/metrics/history')
def get_metrics_history():
    """Obtiene historial de métricas"""
    limit = request.args.get('limit', 50, type=int)
    # TODO: Implementar obtención desde base de datos
    return jsonify({
        'metrics': [],
        'count': 0
    })


@app.route('/api/alerts')
def get_alerts():
    """Obtiene alertas de anomalías"""
    with data_lock:
        return jsonify({
            'alerts': anomaly_alerts[-20:],  # Últimas 20 alertas
            'count': len(anomaly_alerts)
        })


@app.route('/api/predictions')
def get_predictions():
    """Obtiene predicciones de fallos"""
    with data_lock:
        return jsonify({
            'predictions': failure_predictions[-10:],
            'count': len(failure_predictions)
        })


@app.route('/api/logs')
def get_logs():
    """Obtiene logs del sistema"""
    limit = request.args.get('limit', 100, type=int)
    with data_lock:
        return jsonify({
            'logs': system_logs[-limit:],
            'count': len(system_logs)
        })


@app.route('/api/vlans')
def get_vlans():
    """Obtiene información de VLANs"""
    with data_lock:
        vlans = current_metrics.get('vlans', {})
        return jsonify({
            'vlans': vlans,
            'count': len(vlans)
        })


@app.route('/api/topology')
def get_topology():
    """Obtiene topología de red"""
    # Topología simulada
    topology = {
        'nodes': [
            {'id': 'core-switch-1', 'type': 'switch', 'status': 'active'},
            {'id': 'core-switch-2', 'type': 'switch', 'status': 'active'},
            {'id': 'ap-1', 'type': 'access_point', 'status': 'active'},
            {'id': 'ap-2', 'type': 'access_point', 'status': 'active'},
            {'id': 'ap-3', 'type': 'access_point', 'status': 'active'},
            {'id': 'firewall', 'type': 'firewall', 'status': 'active'},
            {'id': 'ids-ips', 'type': 'security', 'status': 'active'}
        ],
        'links': [
            {'source': 'core-switch-1', 'target': 'core-switch-2'},
            {'source': 'core-switch-1', 'target': 'ap-1'},
            {'source': 'core-switch-1', 'target': 'ap-2'},
            {'source': 'core-switch-2', 'target': 'ap-3'},
            {'source': 'core-switch-2', 'target': 'firewall'},
            {'source': 'firewall', 'target': 'ids-ips'}
        ]
    }
    return jsonify(topology)


@socketio.on('connect')
def handle_connect():
    """Maneja conexión de cliente WebSocket"""
    logger.info(f"Cliente conectado: {request.sid}")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Maneja desconexión de cliente"""
    logger.info(f"Cliente desconectado: {request.sid}")


def update_metrics(metrics: Dict[str, Any]):
    """
    Actualiza métricas y notifica a clientes conectados
    
    Args:
        metrics: Diccionario con métricas actualizadas
    """
    global current_metrics
    
    with data_lock:
        current_metrics = metrics
        
        # Añadir timestamp si no existe
        if 'timestamp' not in current_metrics:
            current_metrics['timestamp'] = datetime.now().isoformat()
    
    # Emitir actualización a todos los clientes
    socketio.emit('metrics_update', current_metrics)
    
    # Añadir log
    add_log('INFO', 'Métricas actualizadas')


def add_anomaly_alert(alert: Dict[str, Any]):
    """
    Añade alerta de anomalía
    
    Args:
        alert: Diccionario con información de la alerta
    """
    global anomaly_alerts
    
    with data_lock:
        anomaly_alerts.append(alert)
        
        # Mantener solo últimas 100 alertas
        if len(anomaly_alerts) > 100:
            anomaly_alerts = anomaly_alerts[-100:]
    
    # Emitir alerta a clientes
    socketio.emit('anomaly_alert', alert)
    
    # Log
    severity = alert.get('severity', 'unknown')
    message = alert.get('message', 'Anomalía detectada')
    add_log('WARNING', f"Anomalía {severity}: {message}")


def add_failure_prediction(prediction: Dict[str, Any]):
    """
    Añade predicción de fallo
    
    Args:
        prediction: Diccionario con información de la predicción
    """
    global failure_predictions
    
    with data_lock:
        failure_predictions.append(prediction)
        
        # Mantener solo últimas 50 predicciones
        if len(failure_predictions) > 50:
            failure_predictions = failure_predictions[-50:]
    
    # Emitir predicción a clientes
    socketio.emit('failure_prediction', prediction)
    
    # Log
    if prediction.get('prediction'):
        probability = prediction.get('failure_probability', 0)
        add_log('ERROR', f"Fallo predicho con {probability}% probabilidad")


def add_log(level: str, message: str):
    """
    Añade entrada al log del sistema
    
    Args:
        level: Nivel del log (INFO, WARNING, ERROR, CRITICAL)
        message: Mensaje del log
    """
    global system_logs
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message
    }
    
    with data_lock:
        system_logs.append(log_entry)
        
        # Mantener solo últimos 500 logs
        if len(system_logs) > 500:
            system_logs = system_logs[-500:]
    
    # Emitir log a clientes
    socketio.emit('new_log', log_entry)


def simulate_data_stream():
    """Simula stream de datos para testing (solo para desarrollo)"""
    import numpy as np
    
    while True:
        # Simular métricas
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_packets': np.random.randint(800, 1500),
            'capture_duration': 5.0,
            'latency': {
                'avg_ms': np.random.uniform(10, 40),
                'jitter_ms': np.random.uniform(1, 8),
                'max_ms': np.random.uniform(30, 80),
                'min_ms': np.random.uniform(5, 15)
            },
            'system': {
                'cpu_percent': np.random.uniform(20, 70),
                'memory_percent': np.random.uniform(40, 75),
                'disk_percent': np.random.uniform(30, 60),
                'network_connections': np.random.randint(50, 200)
            },
            'vlans': {
                10: {
                    'name': 'LABORATORIOS',
                    'packets': np.random.randint(200, 400),
                    'bytes': np.random.randint(100000, 500000),
                    'throughput_mbps': np.random.uniform(100, 350),
                    'error_rate': np.random.uniform(0, 2),
                    'protocol_distribution': {
                        'TCP': np.random.uniform(40, 60),
                        'UDP': np.random.uniform(20, 40),
                        'ICMP': np.random.uniform(5, 15),
                        'Other': np.random.uniform(0, 5)
                    }
                },
                20: {
                    'name': 'WIFI_COMUN',
                    'packets': np.random.randint(300, 600),
                    'bytes': np.random.randint(150000, 700000),
                    'throughput_mbps': np.random.uniform(50, 200),
                    'error_rate': np.random.uniform(0, 3),
                    'protocol_distribution': {
                        'TCP': np.random.uniform(35, 55),
                        'UDP': np.random.uniform(25, 45),
                        'ICMP': np.random.uniform(5, 15),
                        'Other': np.random.uniform(0, 5)
                    }
                },
                30: {
                    'name': 'SERVIDORES',
                    'packets': np.random.randint(150, 300),
                    'bytes': np.random.randint(80000, 400000),
                    'throughput_mbps': np.random.uniform(200, 500),
                    'error_rate': np.random.uniform(0, 1),
                    'protocol_distribution': {
                        'TCP': np.random.uniform(60, 80),
                        'UDP': np.random.uniform(10, 25),
                        'ICMP': np.random.uniform(2, 8),
                        'Other': np.random.uniform(0, 3)
                    }
                }
            }
        }
        
        update_metrics(metrics)
        
        # Simular anomalía ocasionalmente (10% probabilidad)
        if np.random.random() < 0.1:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'is_anomaly': True,
                'anomaly_score': np.random.uniform(-0.8, -0.3),
                'confidence': np.random.uniform(70, 95),
                'severity': np.random.choice(['low', 'medium', 'high']),
                'message': 'Anomalía detectada en patrón de tráfico',
                'thresholds_exceeded': []
            }
            
            if metrics['system']['cpu_percent'] > 65:
                alert['thresholds_exceeded'].append(f"CPU alta: {metrics['system']['cpu_percent']:.1f}%")
            
            add_anomaly_alert(alert)
        
        # Simular predicción de fallo ocasionalmente (5% probabilidad)
        if np.random.random() < 0.05:
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'failure_probability': np.random.uniform(50, 95),
                'prediction': True,
                'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                'message': 'Riesgo de fallo detectado',
                'eta': (datetime.now()).isoformat(),
                'confidence': np.random.uniform(60, 90)
            }
            add_failure_prediction(prediction)
        
        time.sleep(5)  # Actualizar cada 5 segundos


def run_dashboard(config: Dict[str, Any], simulation: bool = False):
    """
    Inicia el dashboard web
    
    Args:
        config: Configuración del sistema
        simulation: Si True, simula datos para testing
    """
    dashboard_config = config.get('dashboard', {})
    host = dashboard_config.get('host', '0.0.0.0')
    port = dashboard_config.get('port', 5000)
    debug = dashboard_config.get('debug', False)
    
    logger.info(f"Iniciando dashboard en http://{host}:{port}")
    
    # Si está en modo simulación, iniciar thread de simulación
    if simulation:
        logger.info("Modo simulación activado")
        simulation_thread = threading.Thread(target=simulate_data_stream, daemon=True)
        simulation_thread.start()
    
    # Iniciar servidor
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    # Configuración de prueba
    config = {
        'dashboard': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': True
        }
    }
    
    print("=" * 80)
    print("DASHBOARD DE GESTIÓN DE RED UNIVERSITARIA")
    print("Universidad Nacional de Colombia")
    print("=" * 80)
    print("\nIniciando servidor web...")
    print("Acceda al dashboard en: http://localhost:5000")
    print("Presione Ctrl+C para detener\n")
    
    run_dashboard(config, simulation=True)