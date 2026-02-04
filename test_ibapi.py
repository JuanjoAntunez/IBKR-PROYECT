"""
Test de conexion con TWS usando la API oficial de Interactive Brokers
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time


class IBApp(EWrapper, EClient):
    """Aplicacion simple de IB API."""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.accounts = []
        self.net_liquidation = None
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Manejo de errores."""
        # Filtrar warnings comunes
        if errorCode in [2104, 2106, 2158]:
            print(f"Info: {errorString}")
        elif errorCode >= 2000:
            print(f"Warning [{errorCode}]: {errorString}")
        else:
            print(f"Error [{errorCode}]: {errorString}")
    
    def connectAck(self):
        """Confirmacion de conexion."""
        print("✓ Conexion establecida")
        self.connected = True
    
    def managedAccounts(self, accountsList):
        """Recibe la lista de cuentas."""
        self.accounts = accountsList.split(',')
        print(f"Cuentas disponibles: {self.accounts}")
    
    def accountSummary(self, reqId, account, tag, value, currency):
        """Recibe valores de resumen de cuenta."""
        if tag == "NetLiquidation":
            self.net_liquidation = f"{value} {currency}"
            print(f"Valor de cuenta ({account}): {value} {currency}")


def main():
    """Funcion principal."""
    print("=" * 60)
    print("TEST DE CONEXION CON IB TWS - API OFICIAL")
    print("=" * 60)
    
    # Configuracion
    HOST = '127.0.0.1'
    PORT = 7496  # Cambia a 7497 para Paper Trading
    CLIENT_ID = 1
    
    # Crear aplicacion
    app = IBApp()
    
    try:
        print(f"\nIntentando conectar a {HOST}:{PORT}...")
        
        # Conectar
        app.connect(HOST, PORT, CLIENT_ID)
        
        # Iniciar thread de la API
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()
        
        # Esperar a que se establezca la conexion
        timeout = 10
        start_time = time.time()
        
        while not app.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not app.connected:
            print("\n✗ Timeout: No se pudo conectar")
            print("\nVerifica:")
            print("1. TWS esta abierto y logueado")
            print("2. Puerto correcto (7496=Live, 7497=Paper)")
            print("3. API habilitada en TWS:")
            print("   File → Global Configuration → API → Settings")
            print("   ✓ Enable ActiveX and Socket Clients")
            return
        
        print("\n✓ CONEXION EXITOSA!")
        
        # Esperar a recibir cuentas
        time.sleep(2)
        
        # Solicitar resumen de cuenta si hay cuentas disponibles
        if app.accounts:
            print(f"\nSolicitando resumen de cuenta...")
            app.reqAccountSummary(1, "All", "NetLiquidation")
            time.sleep(3)
        
        print("\n✓ Test completado exitosamente")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        
    finally:
        # Desconectar
        if app.isConnected():
            app.disconnect()
            print("\n✓ Desconectado")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
