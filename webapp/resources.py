"""Consumo de recursos por proceso (= por pantalla), medido desde AFUERA, en el proceso servidor.

Medir desde el propio worker le cobraria al modelo el costo de medirse (como el ResourceSampler de
realtime_pipeline.py, que corre dentro del proceso del benchmark). Aqui un hilo del servidor lee
cada 0.5 s, por PID:
  - CPU % del proceso (psutil; 100 % = un nucleo completo) y su equivalente normalizado a la maquina
  - RAM residente (RSS) e hilos
  - VRAM del proceso via NVML (nvmlDeviceGetComputeRunningProcesses). En Windows con controlador
    WDDM NVML suele no reportar memoria por proceso (usedGpuMemory = None); en ese caso la UI usa
    la VRAM que reporta torch desde el worker YOLO, etiquetada como tal.
y para el sistema completo: CPU, RAM, utilizacion de GPU y VRAM total.
"""
from __future__ import annotations

import threading
import time

import psutil

INTERVAL_S = 0.5


class _Nvml:
    def __init__(self):
        self.handle = None
        self.name = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self.handle)
            self.name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            self.nv = None

    def device(self):
        if self.handle is None:
            return {}
        util = self.nv.nvmlDeviceGetUtilizationRates(self.handle)
        mem = self.nv.nvmlDeviceGetMemoryInfo(self.handle)
        return {"gpu_pct": float(util.gpu), "vram_used_mb": mem.used / 2**20,
                "vram_total_mb": mem.total / 2**20, "gpu_name": self.name}

    def vram_by_pid(self):
        if self.handle is None:
            return {}
        out = {}
        for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                for p in getattr(self.nv, fn)(self.handle):
                    if p.usedGpuMemory is not None:
                        out[p.pid] = out.get(p.pid, 0) + p.usedGpuMemory / 2**20
            except Exception:
                pass
        return out


class ResourceMonitor:
    def __init__(self, on_sample, interval_s: float = INTERVAL_S):
        self.on_sample = on_sample          # callback(dict) en cada muestra
        self.interval_s = interval_s
        self.rows: list[dict] = []          # serie temporal para recursos.csv
        self._pids: dict[str, int] = {}
        self._procs: dict[str, psutil.Process] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._nvml = _Nvml()
        self._ncpu = psutil.cpu_count() or 1
        self._thread = None
        self._t0 = time.perf_counter()

    def track(self, name: str, pid: int) -> None:
        with self._lock:
            self._pids[name] = pid
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(None)  # primera lectura de referencia (siempre 0)
                self._procs[name] = proc
            except psutil.Error:
                pass

    def start(self) -> None:
        psutil.cpu_percent(None)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _sample(self) -> dict:
        t = time.perf_counter() - self._t0
        vram = self._nvml.vram_by_pid()
        procs = {}
        with self._lock:
            items = list(self._procs.items())
        for name, proc in items:
            try:
                with proc.oneshot():
                    cpu = proc.cpu_percent(None)
                    rss = proc.memory_info().rss / 2**20
                    threads = proc.num_threads()
            except psutil.Error:
                continue  # el proceso ya termino
            procs[name] = {"cpu_pct": cpu, "cpu_pct_machine": cpu / self._ncpu, "rss_mb": rss,
                           "threads": threads, "vram_mb": vram.get(proc.pid)}
        vm = psutil.virtual_memory()
        system = {"cpu_pct": psutil.cpu_percent(None), "ram_used_gb": (vm.total - vm.available) / 2**30,
                  "ram_total_gb": vm.total / 2**30, "ram_pct": vm.percent, "cpus": self._ncpu,
                  **self._nvml.device()}
        for name, p in procs.items():
            self.rows.append({"t_s": round(t, 3), "proc": name, **{k: v for k, v in p.items()}})
        self.rows.append({"t_s": round(t, 3), "proc": "sistema", "cpu_pct": system["cpu_pct"],
                          "rss_mb": system["ram_used_gb"] * 1024, "vram_mb": system.get("vram_used_mb"),
                          "gpu_pct": system.get("gpu_pct")})
        return {"t_s": t, "procs": procs, "system": system}

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.on_sample(self._sample())
            except Exception:
                pass  # medir nunca debe tumbar la sesion
