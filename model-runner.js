// model-runner.js — Carga, benchmark y fallback para modelos LLM browser-side
// Soporta modelos multimodales (Gemma 4 E2B) y text-only (Qwen, Phi, SmolLM3).
// Uso:
//   const runner = new ModelRunner({ onProgress, onLog });
//   await runner.load({ modelKey: 'qwen2.5-1.5b', dtype: 'q4f16', device: 'webgpu' });

import { getModel } from './models-registry.js';

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest';

export class ModelRunner {
  constructor({ onProgress, onLog, onFallback, fallbackUrl = null } = {}) {
    this.onProgress = onProgress || (() => {});
    this.onLog = onLog || (() => {});
    this.onFallback = onFallback || (() => {});
    this.fallbackUrl = fallbackUrl;
    this.model = null;
    this.processor = null;
    this.tokenizer = null;
    this.lib = null;
    this.usingFallback = false;
    this.currentModel = null;     // entrada del registry actualmente cargada
    this.dtype = 'q4f16';
  }

  buildDtypeConfig(model, dtype, textOnly) {
    if (typeof dtype !== 'string') return dtype;
    const cfg = {};
    for (const c of model.components) {
      if ((c === 'vision_encoder' || c === 'audio_encoder') && textOnly) {
        cfg[c] = 'q4';
      } else {
        cfg[c] = dtype;
      }
    }
    return cfg;
  }

  async load(opts = {}) {
    const {
      modelKey = 'gemma-4-e2b',
      dtype = 'q4f16',
      device = 'webgpu',
      textOnly = false,
    } = opts;

    const model = getModel(modelKey);
    this.currentModel = model;
    this.dtype = dtype;

    const dtypeConfig = this.buildDtypeConfig(model, dtype, textOnly);

    try {
      this.onLog(`Importando @huggingface/transformers...`);
      this.lib = await import(/* @vite-ignore */ TRANSFORMERS_CDN);

      const { AutoProcessor, AutoTokenizer, AutoModelForCausalLM } = this.lib;

      this.onLog(`Cargando tokenizer de ${model.id}...`);
      this.tokenizer = await AutoTokenizer.from_pretrained(model.id);

      if (model.multimodal) {
        this.onLog(`Cargando processor multimodal...`);
        this.processor = await AutoProcessor.from_pretrained(model.id);
      } else {
        this.processor = null;
      }

      // Selecciona la clase correcta
      let ModelClass;
      if (model.architecture === 'AutoModelForCausalLM') {
        ModelClass = AutoModelForCausalLM;
      } else {
        ModelClass = this.lib[model.architecture];
        if (!ModelClass) {
          throw new Error(`Esta versión de transformers.js no exporta ${model.architecture}`);
        }
      }

      this.onLog(`Descargando modelo · ${model.label} · device=${device} · dtype=${JSON.stringify(dtypeConfig)}`);

      this.model = await ModelClass.from_pretrained(model.id, {
        dtype: dtypeConfig,
        device,
        progress_callback: (p) => this.onProgress(p),
      });

      this.onLog(`Modelo listo: ${model.label}`);
      return { ok: true, model };
    } catch (e) {
      this.onLog(`Error cargando modelo: ${e.message}`);
      if (this.fallbackUrl) {
        this.usingFallback = true;
        this.onFallback({ reason: e.message });
        this.onLog(`Activando fallback server-side: ${this.fallbackUrl}`);
        return { ok: true, fallback: true };
      }
      return { ok: false, error: e };
    }
  }

  async unload() {
    try {
      if (this.model?.dispose) await this.model.dispose();
    } catch {}
    this.model = null;
    this.processor = null;
    this.tokenizer = null;
    this.currentModel = null;
    this.lib = null;
    this.onLog?.('Modelo descargado de memoria.');
  }

  // Construye los inputs del modelo desde mensajes en formato unificado
  // Para multimodal: messages con content array [{type:'text'|'image'|'audio'}]
  // Para text-only: messages con content string o array (extrae solo text)
  async _buildInputs(messages) {
    if (this.currentModel.multimodal && this.processor) {
      const prompt = this.processor.apply_chat_template(messages, {
        add_generation_prompt: true,
        enable_thinking: false,
      });
      return await this.processor(prompt);
    }
    // Text-only: aplana content a string
    const flat = messages.map(m => {
      let content = m.content;
      if (Array.isArray(content)) {
        content = content.filter(c => c.type === 'text').map(c => c.text).join('\n');
      }
      return { role: m.role, content };
    });
    const inputs = this.tokenizer.apply_chat_template(flat, {
      add_generation_prompt: true,
      return_tensor: true,
    });
    // apply_chat_template puede devolver tensor o {input_ids}
    if (inputs?.input_ids) return inputs;
    return { input_ids: inputs };
  }

  async benchmark({ tokens = 32 } = {}) {
    if (this.usingFallback) return { skipped: true, reason: 'fallback activo' };
    if (!this.model) throw new Error('Modelo no cargado');

    const messages = [{
      role: 'user',
      content: [{ type: 'text', text: 'Cuenta del 1 al 30 separado por comas.' }],
    }];

    const inputs = await this._buildInputs(messages);

    this.onLog('Warmup (compilando shaders)...');
    const tWarm = performance.now();
    await this.model.generate({ ...inputs, max_new_tokens: 4, do_sample: false });
    const warmupMs = performance.now() - tWarm;

    let firstTokenMs = null;
    const t0 = performance.now();
    const { TextStreamer } = this.lib;
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      callback_function: () => {
        if (firstTokenMs == null) firstTokenMs = performance.now() - t0;
      },
    });

    const output = await this.model.generate({
      ...inputs,
      max_new_tokens: tokens,
      do_sample: false,
      streamer,
    });
    const totalMs = performance.now() - t0;

    const inputLen = inputs.input_ids?.dims?.[1] ?? inputs.input_ids?.length ?? 0;
    const outputLen = output?.dims?.[1] ?? output?.length ?? 0;
    const generated = Math.max(0, outputLen - inputLen);

    return {
      warmupMs: Math.round(warmupMs),
      ttftMs: Math.round(firstTokenMs ?? totalMs),
      totalMs: Math.round(totalMs),
      tokens: generated || tokens,
      tokensPerSecond: +((generated || tokens) / (totalMs / 1000)).toFixed(2),
      modelLabel: this.currentModel?.label ?? 'desconocido',
    };
  }

  async generate(messages, { maxTokens = 256, onToken } = {}) {
    if (this.usingFallback) return this._generateFallback(messages, { maxTokens, onToken });
    if (!this.model) throw new Error('Modelo no cargado');

    const inputs = await this._buildInputs(messages);

    let text = '';
    const { TextStreamer } = this.lib;
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      callback_function: (chunk) => {
        text += chunk;
        onToken?.(chunk, text);
      },
    });

    const t0 = performance.now();
    await this.model.generate({
      ...inputs,
      max_new_tokens: maxTokens,
      do_sample: true,
      temperature: 1.0,
      top_p: 0.95,
      top_k: 64,
      streamer,
    });
    const ms = performance.now() - t0;

    return { text, latencyMs: Math.round(ms) };
  }

  async _generateFallback(messages, { maxTokens, onToken }) {
    const t0 = performance.now();
    const res = await fetch(this.fallbackUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages, max_tokens: maxTokens }),
    });
    if (!res.ok) throw new Error(`Fallback ${res.status}: ${await res.text()}`);

    if (res.headers.get('content-type')?.includes('text/event-stream') && res.body) {
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let text = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value);
        text += chunk;
        onToken?.(chunk, text);
      }
      return { text, latencyMs: Math.round(performance.now() - t0), via: 'fallback-stream' };
    }

    const data = await res.json();
    const text = data.text ?? data.choices?.[0]?.message?.content ?? '';
    onToken?.(text, text);
    return { text, latencyMs: Math.round(performance.now() - t0), via: 'fallback' };
  }
}
