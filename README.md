# Entorno de Ciencia de Datos en Docker (para usar con VSCode)

Este entorno está pensado para quienes están comenzando en ciencia de datos y quieren un entorno listo para usar sin instalar Python ni librerías manualmente.

## ✅ ¿Qué incluye?

- Python 3.11
- JupyterLab
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, Statsmodels, SciPy
- Plotly, TQDM, PyArrow, OpenPyXL, xlrd

> Todo corre dentro de Docker, ¡sin ensuciar tu máquina!

---

## 🚀 ¿Cómo usarlo desde VSCode?

### 1. Prerrequisitos

Instalá:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Extensiones de VSCode:
  - Remote - Containers
  - Jupyter (opcional)

---

### 2. Pasos para usar

1. **Descomprimir este ZIP**
2. **Abrir la carpeta `mi_entorno_ciencia_datos/` en VSCode**
3. **Cuando VSCode pregunte si querés "Reopen in Container", hacé clic en eso**
4. Esperá que construya el contenedor (puede tardar un poco la primera vez)
5. Abrí el archivo `notebooks/ejemplo_intro.ipynb`
6. ¡Ya podés empezar a jugar con código Python!

---

## 🧪 Alternativa sin VSCode

También podés levantar el entorno desde terminal:

```bash
docker build -t ciencia-datos .
docker run -p 8888:8888 -v "$(pwd)/notebooks:/app/notebooks" ciencia-datos
```

Después abrí el enlace que aparece en consola (es un JupyterLab en el navegador).

---

## 📂 Estructura del proyecto

```
mi_entorno_ciencia_datos/
├── Dockerfile
├── notebooks/
│   └── ejemplo_intro.ipynb
└── .devcontainer/
    └── devcontainer.json
```

---

## 📬 ¿Dudas?

Podés consultarme si algo no te funciona o si querés agregar otras librerías.

¡A programar! 🐍📊
