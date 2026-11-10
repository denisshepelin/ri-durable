(() => {
  const bridge = globalThis.__bridge;
  const emit = globalThis.__emit;
  const config = JSON.parse(globalThis.__config);
  delete globalThis.__bridge;
  delete globalThis.__emit;
  delete globalThis.__config;

  // Host calls: a promise per call, settled by the host through the function this prelude returns.
  const pending = new Map();
  let nextCall = 0;
  const call = (name, args) =>
    new Promise((resolve, reject) => {
      const id = nextCall++;
      pending.set(id, { resolve, reject });
      bridge(id, name, JSON.stringify(args === undefined ? {} : args));
    });
  const settle = (id, ok, json) => {
    const waiting = pending.get(id);
    if (!waiting) return;
    pending.delete(id);
    if (ok) waiting.resolve(JSON.parse(json));
    else waiting.reject(new Error(JSON.parse(json)));
  };

  globalThis.tools = new Proxy(
    {},
    { get: (_, name) => (typeof name === "string" ? (args) => call(name, args) : undefined) },
  );

  const show = (value) => {
    if (typeof value === "string") return value;
    if (value instanceof Error) return `${value.name}: ${value.message}`;
    try {
      const json = JSON.stringify(value, null, 2);
      return json === undefined ? String(value) : json;
    } catch {
      return String(value);
    }
  };
  globalThis.print = (...values) => emit(values.map(show).join(" "));
  globalThis.console = { log: print, info: print, warn: print, error: print, debug: print };

  const catalog = config.catalog;
  globalThis.searchTools = (query) => {
    const words = String(query).toLowerCase().split(/\s+/).filter(Boolean);
    return catalog
      .filter((tool) => {
        const text = `${tool.name} ${tool.description}`.toLowerCase();
        return words.every((word) => text.includes(word));
      })
      .map((tool) => ({ name: tool.name, description: tool.description }));
  };
  globalThis.describeTool = (name) => catalog.find((tool) => tool.name === name)?.declaration;

  // Determinism, so a rerun after a restart takes the same path: seeded randomness and a virtual clock.
  let seed = config.seed | 0;
  Math.random = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const RealDate = Date;
  let now = config.clock;
  globalThis.Date = class extends RealDate {
    constructor(...values) {
      if (values.length === 0) super(now++);
      else super(...values);
    }
    static now() {
      return now++;
    }
  };

  return settle;
})()
