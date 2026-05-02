# Browser LLM Lab

Demo browser-side de inferencia de modelos LLM (Gemma 4 E2B, Qwen 2.5, SmolLM3, Phi-3.5) usando [Transformers.js](https://github.com/huggingface/transformers.js) + WebGPU. Cero servidor, cero backend — toda la inferencia corre en el dispositivo del usuario.

## Qué hace

- **Detecta capacidad** del navegador (WebGPU, RAM, GPU, storage)
- **Carga modelos** ONNX desde Hugging Face Hub con progreso agregado y ETA
- **Benchmarka** velocidad real de inferencia (tok/s, TTFT, warmup)
- **Inferencia** con streaming token a token
- **Gestiona cache** local (Cache API + OPFS), permite borrar versiones por modelo/dtype
- **Multi-modelo**: 5 modelos preconfigurados con velocidades esperadas por GPU

## Modelos soportados

| Modelo | Tamaño q4 | Multimodal | Uso recomendado |
|--------|-----------|------------|-----------------|
| Qwen 2.5 0.5B | ~400 MB | No | Más rápido, decente |
| Qwen 2.5 1.5B | ~1 GB | No | Balance velocidad/calidad |
| SmolLM3 3B | ~2 GB | No | Razonamiento, multilingüe |
| Phi-3.5 mini | ~2.5 GB | No | Razonamiento estructurado |
| Gemma 4 E2B | ~3.4 GB | Sí (texto/imagen/audio) | Calidad, multimodal |

## Requisitos del navegador

- **Chrome/Edge 113+** o **Safari 17+** con WebGPU habilitado
- **8 GB RAM** mínimo, 16 GB recomendado para modelos grandes
- **5+ GB libres** en storage del browser (OPFS) para cachear modelos
- **WebGPU**: GPU dedicada o iGPU moderna (Intel Xe, Apple M1+, AMD RDNA2+)
- Sin WebGPU → fallback a WASM, casi inutilizable para modelos >1B

## Uso local

```bash
# Servir con cualquier HTTP server (HTTPS no requerido para localhost)
python -m http.server 8000
# o
npx serve .
```

Abrir `http://localhost:8000` y seguir el flujo:
1. **Detectar** capacidad → ve si tu GPU soporta el caso
2. **Cargar modelo** → seleccionar dtype/device → primera vez descarga 0.4-3.4 GB
3. **Benchmark** → 32 tokens, mide tok/s real
4. **Inferencia** → escribir prompt y generar
5. **Cache** → ver/borrar modelos descargados

## Deploy en Cloudflare Pages

### Opción A: via dashboard (recomendado)

1. Push este repo a GitHub
2. En [Cloudflare dashboard](https://dash.cloudflare.com) → **Workers & Pages** → **Create application** → **Pages** → **Connect to Git**
3. Seleccionar el repo
4. Configuración:
   - **Build command**: (vacío)
   - **Build output directory**: `/`
   - **Root directory**: `/`
5. Deploy. Pages te asigna `tu-proyecto.pages.dev`

### Opción B: via Wrangler CLI

```bash
npm i -g wrangler
wrangler login
wrangler pages deploy . --project-name=browser-llm-lab
```

### Verificar headers después del deploy

```bash
curl -I https://tu-app.pages.dev/ | grep -iE "cross-origin|opener|embedder"
```

Deberían aparecer:
- `cross-origin-opener-policy: same-origin`
- `cross-origin-embedder-policy: require-corp`

Sin estos, ONNX Runtime Web cae a single-thread y la inferencia es 2-4x más lenta.

## Estructura del proyecto

```
.
├── index.html              # UI principal con 5 paneles
├── capability.js           # Detección WebGPU, GPU, RAM, storage, OPFS
├── model-runner.js         # Carga, benchmark, inferencia, fallback
├── models-registry.js      # Catálogo de modelos disponibles
├── cache-manager.js        # Listar/borrar cache (Cache API + OPFS)
├── _headers                # COOP/COEP para Cloudflare Pages
└── README.md
```

Sin build step, sin dependencias npm, sin bundler. Todo ES modules cargados directo desde el browser.

## Caveats para usuarios

- **Primera carga**: 0.4-3.4 GB de descarga según el modelo. En conexiones lentas, 30-60 minutos.
- **Storage**: cada modelo ocupa su tamaño en el browser (OPFS), persistente entre sesiones.
- **WebGPU obligatorio** para velocidad usable. El panel de capability detecta esto y avisa.
- **Mobile**: posible en flagships recientes, pero las GPUs móviles saturan rápido. UX pobre.
- **Privacidad**: cero data sale del browser. Las inferencias son 100% locales.

## Performance esperada (Gemma 4 E2B q4f16)

| Hardware | tok/s |
|----------|-------|
| Intel iGPU gen-11 | ~1 |
| Intel Arc 140T (Xe2) | 4-8 |
| Apple M1/M2 | 8-15 |
| RTX 3060/4060 | 25-40 |
| RTX 4090 | 60-80 |

Para modelos más chicos (Qwen 0.5B), el speedup vs E2B es solo ~2-3x browser-side por overhead de dispatches WebGPU.

## Optimizaciones opcionales (Chrome flags)

`chrome://flags`:
- `#enable-unsafe-webgpu`
- `#enable-webgpu-developer-features`
- `#enable-experimental-web-platform-features`

Pueden agregar 1.5-2x de velocidad activando subgroups en algunos drivers.

## Mirror de modelos en R2 (opcional, para producción)

Por default los modelos se descargan de Hugging Face Hub. Para apps con mucho tráfico, conviene mirror en R2:

```js
import { env } from '@huggingface/transformers';
env.remoteHost = 'https://models.tu-dominio.com';
env.remotePathTemplate = '{model}/resolve/{revision}/';
```

R2 no cobra egress y el storage es ~$0.015/GB-mes. Para Gemma 4 E2B (3.4 GB): ~$0.05/mes.

## Stack relacionado

Este demo es complementario a:
- [agent-skills](https://github.com/MauricioPerera/agent-skills) — spec para distribuir tools a LLM agents
- [agent-skills-cli](https://github.com/MauricioPerera/agent-skills-cli) — implementación de referencia (Node)
- [just-bash-data](https://github.com/MauricioPerera/just-bash-data) — runtime de tools (db + vec)
- [js-doc-store](https://github.com/MauricioPerera/js-doc-store) — document DB vanilla JS
- [js-vector-store](https://github.com/MauricioPerera/js-vector-store) — vector store con cuantización

## Licencia

MIT
