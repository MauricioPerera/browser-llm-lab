// cache-manager.js — listado y borrado del cache de Transformers.js
// Cubre tanto Cache API (donde transformers.js v3 guarda por defecto) como OPFS.
//
// Uso:
//   const list = await listCachedModels();
//   await deleteCachedDtype('onnx-community/gemma-4-E2B-it-ONNX', 'q4f16');
//   await clearAll();

const HF_HOSTS = ['huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs.hf.co', 'cas-bridge.xethub.hf.co'];

function isHFUrl(url) {
  try { return HF_HOSTS.some(h => new URL(url).host.endsWith(h)); }
  catch { return false; }
}

// Extrae { modelId, file, dtype } de una URL del Hub
// ej: https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data
function parseHFUrl(url) {
  try {
    const u = new URL(url);
    const m = u.pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/[^/]+\/(.+)$/);
    if (!m) return null;
    const modelId = m[1];
    const file = m[2];
    // detecta dtype del nombre del archivo
    let dtype = null;
    const nameMatch = file.match(/_(q4f16|q4|fp16|fp32|quantized|int8|q8|q2f16|q2)\.onnx(_data(_\d+)?)?$/);
    if (nameMatch) dtype = nameMatch[1];
    return { modelId, file, dtype, url };
  } catch { return null; }
}

async function getEntrySize(cache, request) {
  try {
    const res = await cache.match(request);
    if (!res) return 0;
    const len = res.headers.get('content-length');
    if (len) return parseInt(len, 10);
    const blob = await res.clone().blob();
    return blob.size;
  } catch { return 0; }
}

export async function listCachedModels({ withSizes = true } = {}) {
  const groups = new Map();   // key = `${modelId}::${dtype||'misc'}`

  // 1. Cache API (default de transformers.js)
  if ('caches' in self) {
    const names = await caches.keys();
    for (const name of names) {
      const cache = await caches.open(name);
      const reqs = await cache.keys();
      for (const req of reqs) {
        if (!isHFUrl(req.url)) continue;
        const parsed = parseHFUrl(req.url);
        if (!parsed) continue;
        const key = `${parsed.modelId}::${parsed.dtype || 'misc'}`;
        if (!groups.has(key)) {
          groups.set(key, {
            modelId: parsed.modelId,
            dtype: parsed.dtype || 'misc',
            cacheName: name,
            files: [],
            totalBytes: 0,
            source: 'cache-api',
          });
        }
        const g = groups.get(key);
        const size = withSizes ? await getEntrySize(cache, req) : 0;
        g.files.push({ url: req.url, file: parsed.file, size });
        g.totalBytes += size;
      }
    }
  }

  // 2. OPFS (algunas versiones)
  try {
    const root = await navigator.storage.getDirectory?.();
    if (root) {
      await walkOPFS(root, '', groups);
    }
  } catch {}

  return [...groups.values()].sort((a, b) => b.totalBytes - a.totalBytes);
}

async function walkOPFS(dir, path, groups) {
  for await (const [name, handle] of dir.entries()) {
    const fullPath = path ? `${path}/${name}` : name;
    if (handle.kind === 'directory') {
      await walkOPFS(handle, fullPath, groups);
    } else {
      // intenta inferir modelId/dtype desde la ruta
      const m = fullPath.match(/([^/]+\/[^/]+)\/.*?(_(q4f16|q4|fp16|fp32|quantized|int8|q8|q2f16|q2))?\.onnx/);
      if (!m) continue;
      const modelId = m[1];
      const dtype = m[3] || 'misc';
      const key = `${modelId}::${dtype}::opfs`;
      if (!groups.has(key)) {
        groups.set(key, {
          modelId, dtype, cacheName: 'OPFS', files: [], totalBytes: 0, source: 'opfs',
        });
      }
      const g = groups.get(key);
      try {
        const file = await handle.getFile();
        g.files.push({ url: fullPath, file: name, size: file.size });
        g.totalBytes += file.size;
      } catch {}
    }
  }
}

export async function deleteGroup(group) {
  let deleted = 0;
  if (group.source === 'cache-api' && 'caches' in self) {
    const cache = await caches.open(group.cacheName);
    for (const f of group.files) {
      const ok = await cache.delete(f.url);
      if (ok) deleted++;
    }
  } else if (group.source === 'opfs') {
    const root = await navigator.storage.getDirectory();
    for (const f of group.files) {
      try {
        await deleteOPFSPath(root, f.url);
        deleted++;
      } catch {}
    }
  }
  return deleted;
}

async function deleteOPFSPath(root, path) {
  const parts = path.split('/');
  const file = parts.pop();
  let dir = root;
  for (const p of parts) dir = await dir.getDirectoryHandle(p);
  await dir.removeEntry(file);
}

export async function clearAll() {
  let cacheApiDeleted = 0;
  if ('caches' in self) {
    const names = await caches.keys();
    for (const n of names) {
      const ok = await caches.delete(n);
      if (ok) cacheApiDeleted++;
    }
  }
  let opfsDeleted = false;
  try {
    const root = await navigator.storage.getDirectory();
    for await (const [name] of root.entries()) {
      try { await root.removeEntry(name, { recursive: true }); opfsDeleted = true; } catch {}
    }
  } catch {}
  return { cacheApiDeleted, opfsDeleted };
}

export function fmtBytes(b) {
  if (!b) return '0 B';
  const u = ['B','KB','MB','GB'];
  let i = 0; let n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 ? 2 : 1)} ${u[i]}`;
}
