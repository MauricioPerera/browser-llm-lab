// capability.js — Detección de capacidad del navegador para inferencia LLM client-side
// Uso: const report = await checkCapability({ modelSizeGB: 3.4, minBufferGB: 2 });

const GB = 1024 ** 3;

export async function checkCapability(opts = {}) {
  const {
    modelSizeGB = 3.4,    // tamaño descarga
    minBufferGB = 2,      // maxBufferSize mínimo recomendado
    minRamGB = 8,         // navigator.deviceMemory mínimo
  } = opts;

  const report = {
    supported: false,
    tier: 'unsupported',  // unsupported | minimal | viable | recommended
    reasons: [],
    warnings: [],
    gpu: null,
    limits: null,
    system: {},
    storage: {},
  };

  // 1. WebGPU
  if (!('gpu' in navigator)) {
    report.reasons.push('WebGPU no disponible en este navegador');
    return report;
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    report.reasons.push(`requestAdapter falló: ${e.message}`);
    return report;
  }

  if (!adapter) {
    report.reasons.push('No se encontró GPU adapter (driver o GPU no compatible)');
    return report;
  }

  // 2. Info GPU
  const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
  report.gpu = {
    vendor: info.vendor || 'unknown',
    architecture: info.architecture || 'unknown',
    device: info.device || 'unknown',
    description: info.description || 'unknown',
    isFallback: adapter.isFallbackAdapter ?? false,
  };

  if (report.gpu.isFallback) {
    report.warnings.push('GPU es adaptador de fallback (software) — performance será inutilizable');
  }

  // 3. Límites
  const L = adapter.limits;
  report.limits = {
    maxBufferSizeGB: L.maxBufferSize / GB,
    maxStorageBufferBindingSizeGB: L.maxStorageBufferBindingSize / GB,
    maxComputeWorkgroupStorageSizeKB: L.maxComputeWorkgroupStorageSize / 1024,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
  };

  if (report.limits.maxBufferSizeGB < minBufferGB) {
    report.reasons.push(
      `maxBufferSize ${report.limits.maxBufferSizeGB.toFixed(2)} GB < ${minBufferGB} GB requerido — el modelo no cargará`
    );
  }

  // 4. Sistema
  report.system = {
    deviceMemoryGB: navigator.deviceMemory ?? null,
    cores: navigator.hardwareConcurrency ?? null,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
  };

  if (report.system.deviceMemoryGB != null && report.system.deviceMemoryGB < minRamGB) {
    report.warnings.push(
      `RAM reportada ${report.system.deviceMemoryGB} GB < ${minRamGB} GB recomendado (puede crashear durante carga)`
    );
  }

  // 5. Storage
  try {
    const est = await navigator.storage.estimate();
    report.storage = {
      quotaGB: est.quota / GB,
      usageGB: est.usage / GB,
      freeGB: (est.quota - est.usage) / GB,
    };
    if (report.storage.freeGB < modelSizeGB * 1.2) {
      report.warnings.push(
        `Storage libre ${report.storage.freeGB.toFixed(1)} GB insuficiente para cachear modelo de ${modelSizeGB} GB`
      );
    }
  } catch (e) {
    report.warnings.push('No se pudo estimar storage');
  }

  // 6. OPFS (cache moderna de Transformers.js)
  report.storage.opfs = 'storage' in navigator && 'getDirectory' in navigator.storage;

  // 7. Veredicto
  report.supported = report.reasons.length === 0;
  report.tier = computeTier(report);

  return report;
}

function computeTier(r) {
  if (!r.supported) return 'unsupported';
  if (r.gpu?.isFallback) return 'unsupported';

  const v = (r.gpu?.vendor || '').toLowerCase();
  const arch = (r.gpu?.architecture || '').toLowerCase();
  const isDiscrete = /nvidia|amd|apple/.test(v) && !/integrated|intel/.test(arch);
  const ram = r.system?.deviceMemoryGB ?? 0;
  const buf = r.limits?.maxBufferSizeGB ?? 0;

  if (isDiscrete && ram >= 16 && buf >= 4) return 'recommended';
  if (buf >= 2 && ram >= 8) return 'viable';
  return 'minimal';
}

export function formatReport(r) {
  const lines = [];
  lines.push(`Tier: ${r.tier.toUpperCase()}`);
  lines.push(`WebGPU: ${r.supported ? 'OK' : 'NO'}`);
  if (r.gpu) {
    lines.push(`GPU: ${r.gpu.vendor} / ${r.gpu.architecture} ${r.gpu.isFallback ? '(FALLBACK)' : ''}`);
  }
  if (r.limits) {
    lines.push(`maxBuffer: ${r.limits.maxBufferSizeGB.toFixed(2)} GB`);
  }
  if (r.system?.deviceMemoryGB != null) {
    lines.push(`RAM: ${r.system.deviceMemoryGB} GB | cores: ${r.system.cores}`);
  }
  if (r.storage?.freeGB != null) {
    lines.push(`Storage libre: ${r.storage.freeGB.toFixed(1)} GB (OPFS: ${r.storage.opfs ? 'sí' : 'no'})`);
  }
  if (r.reasons.length) lines.push(`BLOQUEOS:\n  - ${r.reasons.join('\n  - ')}`);
  if (r.warnings.length) lines.push(`AVISOS:\n  - ${r.warnings.join('\n  - ')}`);
  return lines.join('\n');
}
