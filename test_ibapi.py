"""
Test de conexión con TWS usando ib_insync.
"""

from ib_insync import IB
import time


def main():
    """Función principal."""
    print("=" * 60)
    print("TEST DE CONEXION CON IB TWS - ib_insync")
    print("=" * 60)

    # Configuración
    HOST = "127.0.0.1"
    PORT = 7496  # Cambia a 7497 para Paper Trading
    CLIENT_ID = 1

    ib = IB()

    try:
        print(f"\nIntentando conectar a {HOST}:{PORT}...")
        ib.connect(HOST, PORT, clientId=CLIENT_ID)

        time.sleep(2)

        if ib.isConnected():
            print("✓ CONEXION EXITOSA!")
            accounts = ib.managedAccounts()
            print(f"Cuentas disponibles: {accounts}")

            # Obtener algunos valores de cuenta
            account_values = ib.accountValues()
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
        if ib.isConnected():
            ib.disconnect()
            print("\n✓ Desconectado")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
