# Configuración personalizada de Jupyter Lab
c = get_config()

# Configuración del servidor
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_root = True
c.ServerApp.token = 'datascience2024'
c.ServerApp.password = ''
c.ServerApp.open_browser = False
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True

# IMPORTANTE: Directorio raíz debe ser /workspace para acceder a notebooks
c.ServerApp.root_dir = '/workspace'

# Configuración de la interfaz
c.LabApp.default_url = '/lab'

# Configuración para resetear workspace
c.LabApp.user_settings_dir = '/workspace/.jupyter/user-settings'
c.LabApp.workspaces_dir = '/workspace/.jupyter/workspaces'

# Configuración de archivos - permitir archivos ocultos
c.ContentsManager.allow_hidden = True

# Configuración de navegación de archivos
c.FileContentsManager.delete_to_trash = False

# Logging
c.Application.log_level = 'INFO'

# Deshabilitar cache problemático
c.ServerApp.use_redirect_file = False
