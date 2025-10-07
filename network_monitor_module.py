"""
Módulo de Monitoreo de Red en Tiempo Real
Universidad Nacional de Colombia - Sistema de Gestión de Red

Este módulo captura y analiza métricas de red en tiempo real:
- Throughput por VLAN
- Latencia inter-VLAN
- Tasa de errores CRC
- Utilización de CPU/memoria
- Distribución de protocolos
"""

import time
import psutil
import logging
from scapy.all import sniff, IP, TCP, UDP, ICMP, Ether
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkMonitor:
    """Monitor de red en tiempo real para captura de métricas"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el monitor de red
        
        Args:
            config: Diccionario con configuración del sistema
        """
        self.config = config
        self.interfaces = config['network']['interfaces']
        self.vlans = config['network']['vlans']
        self.capture_interval = config['monitoring']['capture_interval']
        self.packet_count = config['monitoring']['packet_count']
        
        # Métricas acumuladas
        self.metrics_history = deque(maxlen=1000)
        self.packet_buffer = []
        
        # Contadores por VLAN
        self.vlan_stats = defaultdict(lambda: {
            'packets': 0,
            'bytes': 0,
            'errors': 0,
            'protocols': defaultdict(int)
        })
        
        # Timestamps para cálculo de throughput
        self.last_capture_time = time.time()
        
        logger.info(f"Monitor inicializado para interfaces: {self.interfaces}")
    
    def packet_callback(self, packet):
        """
        Callback para procesar cada paquete capturado
        
        Args:
            packet: Paquete capturado por Scapy
        """
        try:
            self.packet_buffer.append(packet)
            
            # Extraer información básica
            if packet.haslayer(Ether):
                frame_size = len(packet)
                
                # Identificar VLAN (simulado por IP range)
                if packet.haslayer(IP):
                    src_ip = packet[IP].src
                    vlan_id = self._identify_vlan(src_ip)
                    
                    if vlan_id:
                        self.vlan_stats[vlan_id]['packets'] += 1
                        self.vlan_stats[vlan_id]['bytes'] += frame_size
                        
                        # Clasificar protocolo
                        if packet.haslayer(TCP):
                            self.vlan_stats[vlan_id]['protocols']['TCP'] += 1
                        elif packet.haslayer(UDP):
                            self.vlan_stats[vlan_id]['protocols']['UDP'] += 1
                        elif packet.haslayer(ICMP):
                            self.vlan_stats[vlan_id]['protocols']['ICMP'] += 1
                        else:
                            self.vlan_stats[vlan_id]['protocols']['Other'] += 1
        
        except Exception as e:
            logger.error(f"Error procesando paquete: {e}")
    
    def _identify_vlan(self, ip_address: str) -> int:
        """
        Identifica la VLAN basándose en la dirección IP
        
        Args:
            ip_address: Dirección IP del paquete
            
        Returns:
            ID de la VLAN o None
        """
        # Convertir IP a formato comparable
        ip_parts = ip_address.split('.')
        
        # Mapear rangos IP a VLANs
        if ip_parts[0] == '10':
            second_octet = int(ip_parts[1])
            if second_octet == 1:
                return 10  # LABORATORIOS
            elif second_octet == 2:
                return 20  # WIFI_COMUN
            elif second_octet == 3:
                return 30  # SERVIDORES
            elif second_octet == 4:
                return 40  # ADMINISTRACION
            elif second_octet == 5:
                return 50  # GESTION
            elif second_octet == 6:
                return 60  # DMZ
            elif second_octet == 7:
                return 70  # VOIP
        
        return None
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas agregadas del período de captura
        
        Returns:
            Diccionario con métricas calculadas
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_capture_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'capture_duration': time_elapsed,
            'total_packets': len(self.packet_buffer),
            'vlans': {}
        }
        
        # Métricas por VLAN
        for vlan_id, stats in self.vlan_stats.items():
            vlan_name = self.vlans.get(vlan_id, {}).get('name', f'VLAN_{vlan_id}')
            
            # Calcular throughput (Mbps)
            throughput = (stats['bytes'] * 8) / (time_elapsed * 1_000_000) if time_elapsed > 0 else 0
            
            # Calcular distribución de protocolos
            total_proto = sum(stats['protocols'].values())
            proto_distribution = {
                proto: (count / total_proto * 100) if total_proto > 0 else 0
                for proto, count in stats['protocols'].items()
            }
            
            metrics['vlans'][vlan_id] = {
                'name': vlan_name,
                'packets': stats['packets'],
                'bytes': stats['bytes'],
                'throughput_mbps': round(throughput, 2),
                'error_rate': round(stats['errors'] / stats['packets'] * 100, 2) if stats['packets'] > 0 else 0,
                'protocol_distribution': proto_distribution
            }
        
        # Métricas del sistema
        metrics['system'] = self._get_system_metrics()
        
        # Calcular latencia estimada (basado en paquetes TCP)
        metrics['latency'] = self._estimate_latency()
        
        # Resetear contadores
        self.last_capture_time = current_time
        self.packet_buffer.clear()
        for vlan_stats in self.vlan_stats.values():
            vlan_stats['packets'] = 0
            vlan_stats['bytes'] = 0
            vlan_stats['errors'] = 0
            vlan_stats['protocols'].clear()
        
        # Guardar en historial
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """
        Obtiene métricas del sistema operativo
        
        Returns:
            Diccionario con métricas del sistema
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections())
        }
    
    def _estimate_latency(self) -> Dict[str, float]:
        """
        Estima latencia basándose en paquetes TCP
        
        Returns:
            Diccionario con métricas de latencia
        """
        tcp_packets = [p for p in self.packet_buffer if p.haslayer(TCP)]
        
        if len(tcp_packets) < 2:
            return {
                'avg_ms': 0,
                'min_ms': 0,
                'max_ms': 0,
                'jitter_ms': 0
            }
        
        # Simulación de latencia basada en tiempos de llegada
        timestamps = [float(p.time) for p in tcp_packets]
        intervals = np.diff(timestamps) * 1000  # Convertir a ms
        
        return {
            'avg_ms': round(np.mean(intervals), 2),
            'min_ms': round(np.min(intervals), 2),
            'max_ms': round(np.max(intervals), 2),
            'jitter_ms': round(np.std(intervals), 2)
        }
    
    def start_monitoring(self, interface: str = None, duration: int = 60):
        """
        Inicia el monitoreo de red
        
        Args:
            interface: Interfaz de red a monitorear (None = todas)
            duration: Duración del monitoreo en segundos
        """
        if interface is None:
            interface = self.interfaces[0]
        
        logger.info(f"Iniciando captura en interfaz {interface} por {duration} segundos")
        
        try:
            # Capturar paquetes
            sniff(
                iface=interface,
                prn=self.packet_callback,
                timeout=duration,
                store=False
            )
            
            # Calcular métricas finales
            final_metrics = self.calculate_metrics()
            logger.info("Captura completada")
            
            return final_metrics
        
        except PermissionError:
            logger.error("Error: Se requieren privilegios de administrador para capturar paquetes")
            return None
        except Exception as e:
            logger.error(f"Error durante la captura: {e}")
            return None
    
    def get_metrics_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """
        Obtiene resumen de las últimas N capturas
        
        Args:
            last_n: Número de capturas a incluir en el resumen
            
        Returns:
            Diccionario con resumen de métricas
        """
        if len(self.metrics_history) == 0:
            return {'error': 'No hay métricas disponibles'}
        
        recent_metrics = list(self.metrics_history)[-last_n:]
        
        # Calcular promedios
        summary = {
            'period': {
                'start': recent_metrics[0]['timestamp'],
                'end': recent_metrics[-1]['timestamp'],
                'captures': len(recent_metrics)
            },
            'averages': {}
        }
        
        # Promediar throughput por VLAN
        for vlan_id in self.vlans.keys():
            throughputs = []
            for metric in recent_metrics:
                if vlan_id in metric.get('vlans', {}):
                    throughputs.append(metric['vlans'][vlan_id]['throughput_mbps'])
            
            if throughputs:
                summary['averages'][f'vlan_{vlan_id}_throughput_mbps'] = round(np.mean(throughputs), 2)
        
        return summary


def main():
    """Función principal para pruebas"""
    # Configuración de prueba
    config = {
        'network': {
            'interfaces': ['eth0'],
            'vlans': {
                10: {'name': 'LABORATORIOS', 'network': '10.1.0.0/16'},
                20: {'name': 'WIFI_COMUN', 'network': '10.2.0.0/16'},
                30: {'name': 'SERVIDORES', 'network': '10.3.0.0/24'}
            }
        },
        'monitoring': {
            'capture_interval': 5,
            'packet_count': 1000
        }
    }
    
    # Crear monitor
    monitor = NetworkMonitor(config)
    
    # Iniciar captura de 30 segundos
    print("Iniciando monitoreo de red...")
    metrics = monitor.start_monitoring(duration=30)
    
    if metrics:
        print("\n" + "=" * 80)
        print("MÉTRICAS CAPTURADAS")
        print("=" * 80)
        print(f"Timestamp: {metrics['timestamp']}")
        print(f"Total de paquetes: {metrics['total_packets']}")
        print(f"\nMétricas del sistema:")
        print(f"  CPU: {metrics['system']['cpu_percent']}%")
        print(f"  Memoria: {metrics['system']['memory_percent']}%")
        print(f"\nLatencia:")
        print(f"  Promedio: {metrics['latency']['avg_ms']} ms")
        print(f"  Jitter: {metrics['latency']['jitter_ms']} ms")
        print("\nMétricas por VLAN:")
        for vlan_id, vlan_data in metrics['vlans'].items():
            print(f"\n  VLAN {vlan_id} ({vlan_data['name']}):")
            print(f"    Paquetes: {vlan_data['packets']}")
            print(f"    Throughput: {vlan_data['throughput_mbps']} Mbps")
            print(f"    Protocolos: {vlan_data['protocol_distribution']}")


if __name__ == "__main__":
    main()