"""
Test simple de conexion con Interactive Brokers TWS
"""

from ib_insync import IB
import time

# Configuracion
HOST = '127.0.0.1'
PORT = 7496  # Cambia a 7497 si usas Paper Trading
CLIENT_ID = 1

print("=" * 50)
print("TEST DE CONEXION CON IB TWS")
print("=" * 50)

# Crear cliente
ib = IB()

try:
    print(f"\nIntentando conectar a {HOST}:{PORT}...")
    
    # Conectar
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    
    # Esperar
    time.sleep(2)
    
    # Verificar
    if ib.isConnected():
        print("✓ CONEXION EXITOSA!")
        
        # Obtener info basica
        print(f"\nCuentas disponibles: {ib.managedAccounts()}")
        
        # Obtener algunos valores de cuenta
        account_values = ib.accountValues()
        print(f"\nTotal de valores de cuenta: {len(account_values)}")
        
        # Mostrar NetLiquidation (valor de cuenta)
        for av in account_values:
            if av.tag == "NetLiquidation":
                print(f"Valor de cuenta: {av.value} {av.currency}")
        
        print("\n✓ Test completado exitosamente")
        
    else:
        print("✗ No se pudo conectar")
        
except ConnectionRefusedError:
    print("\n✗ ERROR: Conexion rechazada")
    print("\nVerifica:")
    print("1. TWS esta abierto y logueado")
    print("2. Puerto correcto (7496=Live, 7497=Paper)")
    print("3. API habilitada en TWS:")
    print("   File → Global Configuration → API → Settings")
    print("   ✓ Enable ActiveX and Socket Clients")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    
finally:
    # Desconectar
    if ib.isConnected():
        ib.disconnect()
        print("\n✓ Desconectado")
    
    print("\n" + "=" * 50)
