// models-registry.js — catálogo de modelos disponibles para el runner
// Cada entrada describe cómo cargarlo y operarlo.
//
// NOTA sobre expectedSpeed: son estimaciones aproximadas. La realidad depende
// del driver, browser, contexto y overhead de WebGPU dispatches. Mediciones
// reales en gen-11 muestran que el speedup entre modelos chicos y grandes
// es menos lineal de lo esperado (Qwen 0.5B → 2 t/s, Gemma 4 E2B → 1 t/s).
// Usar como orden de magnitud, no valores exactos.

export const MODELS = {
  'gemma-4-e2b': {
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    label: 'Gemma 4 E2B — multimodal (texto/imagen/audio)',
    sizeNote: '~3.4 GB (q4f16)',
    multimodal: true,
    architecture: 'Gemma4ForConditionalGeneration',
    components: ['embed_tokens', 'decoder_model_merged', 'vision_encoder', 'audio_encoder'],
    dtypes: ['q4f16', 'q4', 'quantized', 'fp16'],
    expectedSpeed: { 'gen-11-igpu': 1, 'arc-140t': 5, 'rtx-4070': 30 },
  },
  'qwen2.5-1.5b': {
    id: 'onnx-community/Qwen2.5-1.5B-Instruct',
    label: 'Qwen 2.5 1.5B Instruct — texto rápido',
    sizeNote: '~1 GB (q4)',
    multimodal: false,
    architecture: 'AutoModelForCausalLM',
    components: ['model'],
    dtypes: ['q4f16', 'q4', 'quantized', 'fp16'],
    expectedSpeed: { 'gen-11-igpu': 1.5, 'arc-140t': 12, 'rtx-4070': 60 },
  },
  'qwen2.5-0.5b': {
    id: 'onnx-community/Qwen2.5-0.5B-Instruct',
    label: 'Qwen 2.5 0.5B Instruct — ultra rápido',
    sizeNote: '~400 MB (q4)',
    multimodal: false,
    architecture: 'AutoModelForCausalLM',
    components: ['model'],
    dtypes: ['q4f16', 'q4', 'quantized', 'fp16'],
    expectedSpeed: { 'gen-11-igpu': 2, 'arc-140t': 20, 'rtx-4070': 100 },
  },
  'smollm3-3b': {
    id: 'HuggingFaceTB/SmolLM3-3B-ONNX',
    label: 'SmolLM3 3B — texto, balanceado',
    sizeNote: '~2 GB (q4)',
    multimodal: false,
    architecture: 'AutoModelForCausalLM',
    components: ['model'],
    dtypes: ['q4f16', 'q4', 'quantized'],
    expectedSpeed: { 'gen-11-igpu': 1.2, 'arc-140t': 8, 'rtx-4070': 40 },
  },
  'phi-3.5-mini': {
    id: 'onnx-community/Phi-3.5-mini-instruct-onnx-web',
    label: 'Phi-3.5 mini — razonamiento',
    sizeNote: '~2.5 GB (q4)',
    multimodal: false,
    architecture: 'AutoModelForCausalLM',
    components: ['model'],
    dtypes: ['q4f16', 'q4', 'quantized'],
    expectedSpeed: { 'gen-11-igpu': 1.2, 'arc-140t': 7, 'rtx-4070': 35 },
  },
};

export function getModel(key) {
  const m = MODELS[key];
  if (!m) throw new Error(`Modelo desconocido: ${key}`);
  return m;
}
