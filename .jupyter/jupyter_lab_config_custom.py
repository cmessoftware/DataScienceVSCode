# Configuración personalizada para Jupyter Lab
# Esto asegura que notebooks_clean sea el directorio principal

c = get_config()

# Directorio raíz para notebooks
c.ServerApp.root_dir = '/workspace/notebooks_final'

# Configuraciones de seguridad
c.ServerApp.token = 'datascience2024'
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True

# Configuraciones de acceso
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Configuraciones de interfaz
c.LabApp.default_url = '/lab'
